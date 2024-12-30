from __future__ import annotations

from typing import AsyncIterator
import aiomqtt
import json
from rich.console import Console

console = Console()


class MqttClient:
    """MQTT client for collecting measurements from Shelly devices."""

    def __init__(self, host: str, username: str, password: str, topic: str):
        """Initialize the MQTT client.

        The Shelly device must have the MQTT plugin enabled and connected to existing MQTT broker.

        If you don't have a broker, you can use Home Assistant addon.

        Args:
            host: Hostname of the MQTT broker.
            username: Username for the MQTT broker.
            password: Password for the MQTT broker.
            topic: Topic to which the Shelly writes the measurements (e.g. "shellypro3em-mqtt/status/em:0")
        """
        self.hostname = host
        self.username = username
        self.password = password
        self.topic = topic

    async def __aenter__(self) -> MqttClient:
        self.client = aiomqtt.Client(
            hostname=self.hostname,
            username=self.username,
            password=self.password,
        )
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.client.__aexit__(exc_type, exc_value, traceback)

    async def collect_measurements(self) -> AsyncIterator[dict]:
        """Collect measurements from the Shelly device.

        This function subscribes to the MQTT topic and yields the measurements as they come.
        """
        await self.client.subscribe(self.topic)
        async for message in self.client.messages:
            message_data = json.loads(message.payload.decode("utf-8"))
            console.print(message_data)
            yield message_data
