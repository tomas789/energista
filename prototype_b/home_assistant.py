from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

import websockets
from rich.console import Console
from websockets.asyncio.client import connect

console = Console()


class HomeAssistantState:
    """Fetch device states from Home Assistant.

    This uses Home Assistant's websocket API. It first retrieves the initial states and then
    subscribes to the state changes.

    The states are stored and the EKF uses them to estimate the device consumption.
    """

    def __init__(self, api_url: str, api_token: str, entity_ids: list[str]) -> None:
        """Initialize the Home Assistant state collector.

        Args:
            api_url: URL of the Home Assistant API.
            api_token: Token for the Home Assistant API.
            entity_ids: List of entity IDs to collect states for.
        """
        self.api_url = api_url
        self.api_token = api_token
        self.entity_ids = entity_ids

        self.states: OrderedDict[str, Any] = OrderedDict()

        self.ws: websockets.WebSocketClientProtocol | None = None

    async def __aenter__(self) -> HomeAssistantState:
        self.ws = await connect(f"ws://{self.api_url}/api/websocket")
        intro_message = json.loads(await self.ws.recv())
        console.print(intro_message)
        if intro_message["type"] != "auth_required":
            raise ValueError("Authentication failed")
        await self.ws.send(json.dumps({"type": "auth", "access_token": self.api_token}))
        auth_response = json.loads(await self.ws.recv())
        console.print(auth_response)
        return self

    async def _retrieve_initial_states(self) -> None:
        await self.ws.send(json.dumps({"type": "get_states", "id": 1}))
        response = json.loads(await self.ws.recv())
        if response["id"] != 1:
            raise ValueError("Failed to get states")
        if not response["success"]:
            raise ValueError("Failed to get states")
        initial_states = response["result"]
        for state in initial_states:
            if state["entity_id"] not in self.entity_ids:
                continue
            self.states[state["entity_id"]] = state["state"]

        console.print(self.states)

    async def _subscribe_to_states(self) -> None:
        await self.ws.send(json.dumps({"type": "subscribe_events", "id": 2}))
        response = json.loads(await self.ws.recv())
        if response["id"] != 2:
            raise ValueError("Failed to subscribe to events")
        if not response["success"]:
            raise ValueError("Failed to subscribe to events")
        console.print(response)

    async def collect_states(self) -> None:
        await self._retrieve_initial_states()
        await self._subscribe_to_states()

        while True:
            message_str = await self.ws.recv()
            message = json.loads(message_str)
            if message["type"] != "event":
                continue

            event = message["event"]
            event_type = event["event_type"]
            if event_type != "state_changed":
                continue
            event_data = event["data"]
            entity_id = event_data["entity_id"]
            new_state = event_data["new_state"]

            if entity_id not in self.states:
                continue

            self.states[entity_id] = new_state["state"]

            console.print(self.states)

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.ws.close()
