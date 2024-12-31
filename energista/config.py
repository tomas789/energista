import os
import re

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_raw_as


class Ekf(BaseModel):
    """Configuration of the `EKF` class."""

    max_residual: float
    """Maximum acceptable residual for each group [W]."""


class Device(BaseModel):
    entity_id: str
    group: str
    consumption_on: str
    """Consumption when the device is on [W].

    Example: `10 ± 3`
    """

    consumption_off: str
    """Consumption when the device is off [W].

    Example: `0 ± 0.01`
    """


class Measurement(BaseModel):
    """Configuration of the `MqttClient` class."""

    mqtt_host: str
    """MQTT host."""
    mqtt_username: str
    """MQTT username.

    To load from the environment variable, use `${ENV_VAR_NAME}`.
    """
    mqtt_password: str
    """MQTT password.

    To load from the environment variable, use `${ENV_VAR_NAME}`.
    """
    mqtt_topic: str
    """MQTT topic."""


class Group(BaseModel):
    """Configuration of the `Group` class."""

    name: str
    """Name of the group."""

    measurement_field: str
    """Measurement field from Shelly device."""

    residual: bool
    """Whether to use residual in the EKF."""


class HomeAssistant(BaseModel):
    """Configuration of the `HomeAssistant` class."""

    api_url: str
    """Home Assistant API URL."""
    api_token: str
    """Home Assistant API token.

    To load from the environment variable, use `${ENV_VAR_NAME}`.
    """


class Config(BaseModel):
    """The `device.yml` configuration."""

    devices: list[Device]
    """Devices to be monitored."""

    measurement: Measurement
    """MQTT configuration."""

    groups: list[Group]
    """Groups of devices."""

    home_assistant: HomeAssistant
    """Home Assistant configuration."""

    ekf: Ekf


def load_config(path: str) -> Config:
    """Load the configuration from the given path.

    The configuration that should be loaded from environment variables is also resolved here.
    """
    with open(path, "r") as f:
        config = parse_yaml_raw_as(Config, f.read())

    mqtt_username = config.measurement.mqtt_username
    if mqtt_username.startswith("${") and mqtt_username.endswith("}"):
        mqtt_username_env = os.getenv(mqtt_username[2:-1])
        if mqtt_username_env is None:
            raise ValueError(f"MQTT username environment variable {mqtt_username[2:-1]} is not set")
        config.measurement.mqtt_username = mqtt_username_env

    mqtt_password = config.measurement.mqtt_password
    if mqtt_password.startswith("${") and mqtt_password.endswith("}"):
        mqtt_password_env = os.getenv(mqtt_password[2:-1])
        if mqtt_password_env is None:
            raise ValueError(f"MQTT password environment variable {mqtt_password[2:-1]} is not set")
        config.measurement.mqtt_password = mqtt_password_env

    home_assistant_api_token = config.home_assistant.api_token
    if home_assistant_api_token.startswith("${") and home_assistant_api_token.endswith("}"):
        home_assistant_api_token_env = os.getenv(home_assistant_api_token[2:-1])
        if home_assistant_api_token_env is None:
            raise ValueError(f"Home Assistant API token environment variable {home_assistant_api_token[2:-1]} is not set")
        config.home_assistant.api_token = home_assistant_api_token_env

    return config
