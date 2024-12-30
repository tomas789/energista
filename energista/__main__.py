import argparse
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

import autograd.numpy as np
from autograd import jacobian
from pydantic import BaseModel
from rich.console import Console

from . import config as c
from .home_assistant import HomeAssistantState
from .mqtt import MqttClient


load_dotenv()

console = Console()


BANNER = """
[bold green]Prototype B[/bold green]

[italic]
This script estimates consumption of individual devices based on 
their state from Home Assistant and total power measurement 
from Shelly Pro 3EM.
[/italic]
"""


@dataclass
class DeviceInfo:
    """Device info is a structure that holds 'learned' parameters of the device.

    Those parameters are used by the EKF to estimate the device consumption given its state.

    Note that the device info does not hold all the information that the EKF estimates.
    It also estimates the covariances between all the parameters. It is ok to ignore thiese
    covariances as they should be very small. That would basically mean that consumption
    of one device depends (partially) on the state of other devices.
    """

    entity_id: str
    """Entity ID of the device (in Home Assistant)"""

    consumption_on: float
    """Power consumption when device is on [W]"""

    consumption_on_error: float
    """Certainty of power consumption when device is on. [W]

    The consumption estimate has normal distribution:
        N(consumption_on, consumption_on_error^2)
    """

    consumption_off: float
    """Power consumption when device is off [W]"""

    consumption_off_error: float
    """Certainty of power consumption when device is off. [W]

    The consumption estimate has normal distribution:
        N(consumption_off, consumption_off_error^2)
    """


class DeviceInfosFile(BaseModel):
    """File that holds device info.

    This is only used to dump and load device info to a file.
    """

    device_infos: list[DeviceInfo]


class Ekf:
    """Extended Kalman Filter for estimating device consumption."""

    def __init__(self, config: c.Config, device_infos: list[DeviceInfo], device_infos_path: Path):
        """Initialize the EKF.

        Args:
            config: Configuration of the system.
            device_infos: List of device info.
            device_infos_path: Path to which the current estimate of the device info is dumped.
        """
        self.config = config
        self.device_infos = device_infos
        self.device_infos_path = device_infos_path

        self.n_devices = len(self.config.devices)
        self.n_groups = len(self.config.groups)
        self.state_size = 3 * self.n_devices + self.n_groups
        """
        Each device has 3 parameters:
        - device state (0 or 1)
        - device consumption when on
        - device consumption when off

        Each group has 1 parameter:
        - group residual consumption
        """

        x_0 = np.zeros(self.state_size)
        for i, device_info in enumerate(self.device_infos):
            x_0[i * 3 : (i + 1) * 3] = [
                0.0,  # Initial device state. It does not matter.
                device_info.consumption_on,
                device_info.consumption_off,
            ]

        P_0_diag = np.zeros(self.state_size)
        for i, device_info in enumerate(self.device_infos):
            P_0_diag[i * 3 : (i + 1) * 3] = [
                0.01,
                device_info.consumption_on_error,
                device_info.consumption_off_error,
            ]
        for i, group in enumerate(self.config.groups):
            P_0_diag[3 * self.n_devices + i] = 300

        P_0 = np.diag(P_0_diag)

        self.Q: np.ndarray = np.diag((3 * self.n_devices) * [0.01] + self.n_groups * [300])
        """Process noise covariance matrix"""

        self.x_k1: np.ndarray = x_0
        """Previous state estimate"""

        self.P_k1: np.ndarray = P_0
        """Previous state covariance estimate"""

        assert self.x_k1.shape == (self.state_size,)
        assert self.P_k1.shape == (self.state_size, self.state_size)
        assert self.Q.shape == (self.state_size, self.state_size)

        group_name_to_index = {g.name: i for i, g in enumerate(self.config.groups)}

        self.device_i_to_group_i = {i: group_name_to_index[d.group] for i, d in enumerate(self.config.devices)}

    def ekf_step(self, measurement: dict, states: OrderedDict[str, float]) -> None:
        """Perform one step of the EKF.

        This is the place where magic happens.

        First, the 'command' vector `u_k` is created. It contains 1 for each device that is on and 0 for each device that is off.

        Then, the 'measurement' vector `z_k` is created. It contains the total power consumption of each group.

        The function `f` is the state transition function. It propagates the state from the previous step to the next step.
        In our case it is almost an identity function. The only thing it does is that it changes the state of the device
        to the command `u_k`.

        The function `h` is the measurement function. It creates the measurement vector `z_k` from the state.
        It is a function that has the state vector `x_k` as an argument and based on that it calculates the consumption 
        of each group we can expect. It is a 'prediction' we make for the measurement we already have (`measurement`).

        The rest is just the Kalman filter magic.

        Args:
            measurement: Measurement from the Shelly device.
            states: States of the devices from Home Assistant.

        """
        # Observation for Kalman filter
        u_k = np.array([(1.0 if states[d.entity_id] == "on" else 0.0) for d in self.config.devices])
        z_k = np.array([measurement[g.measurement_field] for g in self.config.groups])

        def f(x_k1: np.ndarray, u_k: np.ndarray) -> np.ndarray:
            x_k = [x_k1[i] for i in range(self.state_size)]
            for i, device_info in enumerate(self.device_infos):
                x_k[i * 3] = u_k[i]  # Only change device state. The rest is propagated from previous state.
            return np.array(x_k)

        def h(x_k: np.ndarray) -> np.ndarray:
            z_k = self.n_groups * [0.0]

            for i in range(self.n_devices):
                device_state = x_k[3 * i] > 0.5
                device_consumption_on = x_k[3 * i + 1]
                device_consumption_off = x_k[3 * i + 2]
                device_consumption = device_consumption_on if device_state else device_consumption_off
                print(f"Device {self.config.devices[i].entity_id} has {self.device_i_to_group_i[i]}th group")
                z_k[self.device_i_to_group_i[i]] += device_consumption

            for i in range(self.n_groups):
                z_k[i] += x_k[3 * self.n_devices + i]

            return np.array(z_k)

        # Prediction step
        x_k_pred = f(self.x_k1, u_k)
        J_f = jacobian(lambda x: f(x, u_k))(self.x_k1)
        P_k_pred = J_f @ self.P_k1 @ J_f.T + self.Q

        # Update step
        R = np.eye(self.n_groups) * 5  # Observation noise
        J_h = jacobian(h)(x_k_pred)
        K = P_k_pred @ J_h.T @ np.linalg.inv(J_h @ P_k_pred @ J_h.T + R)
        z_k_pred = h(x_k_pred)
        x_k1 = x_k_pred + K @ (z_k - z_k_pred)
        P_k1 = (np.eye(self.state_size) - K @ J_h) @ P_k_pred

        self.x_k1 = x_k1
        self.P_k1 = P_k1

        self._dump_device_infos()
        self._print_ekf_step(z_k, z_k_pred)

    def _dump_device_infos(self) -> None:
        """Dump the current estimate of the device info to a file.

        The file is then used by `load_device_infos` which is then fed to the EKF to create 
        the initial state of next EKF run (i.e. after program restart).
        """
        device_infos = []
        for i, device in enumerate(self.config.devices):
            device_info = DeviceInfo(
                entity_id=device.entity_id,
                consumption_on=self.x_k1[3 * i + 1],
                consumption_on_error=self.P_k1[3 * i + 1, 3 * i + 1],
                consumption_off=self.x_k1[3 * i + 2],
                consumption_off_error=self.P_k1[3 * i + 2, 3 * i + 2],
            )
            device_infos.append(device_info)
        model = DeviceInfosFile(device_infos=device_infos)
        self.device_infos_path.write_text(model.model_dump_json(indent=2))

    def _print_ekf_step(self, z_k: np.ndarray, z_k_pred: np.ndarray) -> None:
        """Print the EKF step results.

        Args:
            z_k: Measurement.
            z_k_pred: Prediction of the measurement.
        """

        warnings = []

        console.print("[bold green]**********************[/bold green]")
        console.print("[bold green]****** EKF step ******[/bold green]")
        console.print("[bold green]**********************[/bold green]")
        console.print()

        console.print(np.vstack([z_k.reshape(1, -1), z_k_pred.reshape(1, -1)]))

        console.print("[bold green]EKF state[/bold green]")
        console.print("Devices:")
        for i, device_info in enumerate(self.device_infos):
            console.print(f" - [orange bold]{device_info.entity_id}[/orange bold]:")
            console.print(f"     state: {self.x_k1[3 * i]}")
            console.print(f"     consumption on: {self.x_k1[3 * i + 1]}")
            console.print(f"     consumption off: {self.x_k1[3 * i + 2]}")
        console.print("Group residuals:")
        for i, group in enumerate(self.config.groups):
            residual = np.abs(self.x_k1[3 * self.n_devices + i])
            is_high_residual = residual > self.config.ekf.max_residual
            c = "red" if is_high_residual else "green"
            console.print(f" - [orange bold]{group.name}[/orange bold]:")
            console.print(f"     residual: [{c}]{residual}[/{c}]")

            if is_high_residual:
                warnings.append(f"High residual in group {group.name}")
        console.print()

        console.print("[bold green]Measurements[/bold green]")
        for i, group in enumerate(self.config.groups):
            console.print(f" - [orange bold]{group.name}[/orange bold]:")
            console.print(f"     measurement: {z_k[i]}")
            console.print(f"     prediction: {z_k_pred[i]}")
            console.print(f"     innovation: {z_k[i] - z_k_pred[i]}")
        console.print()

        if warnings:
            console.print("[bold red]Warnings:[/bold red]")
            for warning in warnings:
                console.print(f" - {warning}")
            console.print()

    async def run(self, ha: HomeAssistantState, mqtt: MqttClient):
        while True:
            async for message in mqtt.collect_measurements():
                self.ekf_step(message, ha.states)


def load_device_infos(config: c.Config, device_infos_path: Path) -> list[DeviceInfo]:
    """Load device info from a file or create one using default parameters from devices.yml.

    The `device_infos_path` file is created by `Ekf._dump_device_infos`.

    Args:
        config: Configuration of the system.
        device_infos_path: Path to which the current estimate of the device info is dumped.

    Returns:
        List of device info. Each device from devices.yml is represented by one device info.

    """
    existing_device_infos = {}
    if device_infos_path.exists():
        device_infos_file = DeviceInfosFile.model_validate_json(device_infos_path.read_text())
        existing_device_infos = {d.entity_id: d for d in device_infos_file.device_infos}

    device_infos = []
    for d in config.devices:
        device_info = existing_device_infos.get(d.entity_id)
        if device_info is None:
            on, on_err = tuple(d.consumption_on.split("±", maxsplit=1))
            off, off_err = tuple(d.consumption_off.split("±", maxsplit=1))
            device_info = DeviceInfo(
                entity_id=d.entity_id,
                consumption_on=float(on),
                consumption_on_error=float(on_err),
                consumption_off=float(off),
                consumption_off_error=float(off_err),
            )
        device_infos.append(device_info)
    return device_infos


async def main():
    console.print(BANNER)

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    console.print(f"[blue]Loading config from {args.config}[/blue]")
    config = c.load_config(args.config)
    console.print(config)

    entity_ids = [d.entity_id for d in config.devices]
    device_infos_path = Path("device_infos.json")
    device_infos = load_device_infos(config, device_infos_path)

    ha = HomeAssistantState(config.home_assistant.api_url, config.home_assistant.api_token, entity_ids)
    measurement = config.measurement
    mqtt = MqttClient(
        measurement.mqtt_host,
        measurement.mqtt_username,
        measurement.mqtt_password,
        measurement.mqtt_topic,
    )
    ekf = Ekf(config, device_infos, device_infos_path)

    async with ha, mqtt:
        await asyncio.gather(
            ha.collect_states(),
            ekf.run(ha, mqtt),
        )


if __name__ == "__main__":
    asyncio.run(main())
