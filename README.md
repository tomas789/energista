# Energista

Estimate power consumption of all your devices at home without measuring them directly.

I have a **Home Assistant** running at home. It knows the state of most of the energy consuming devices.

I also have a real-time power meter that can measure the power consumption of some devices.

By combining those two pieces of information, I felt like it should be possible to estimate the power consumption of all devices at home.

This is my current work-in-progress.  

## How it works

It reads the state of all devices from Home Assistant and the power consumption from the power meter.

You then provide a "map" of your house wiring (see `devices.yml` for an example).

It uses the Extended Kalman Filter to estimate the power consumption of all devices.

After each measurement from the power meter is received, it updates the state of the devices and estimates the power consumption of all devices and prints it to the console.

## Does it work?

In simulation, it works pretty well (see `prototype.py`).

In practice, I have yet to create a full map of my house wiring. The filter itself seems to work, but the results are not very good.

### Device consumption model

At the moment, the device consumption model is very simple. It just estimates the power consumption when the device is on and when it is off. It works well in many situations but we can do better.

For example, we know dimming state of some light bulbs in Home Assistant. We could use that information to estimate the power consumption of the light bulbs more accurately.

Another example whould be a device where we measure its power consumption but we don't have its state in Home Assistant. In my case, it is a fridge. I measure it directly (smart plug) and I don't have its state in Home Assistant.

## Future work

- [x] Live-update the device states from Home Assistant.
- [x] Live-update the power meter measurements.
- [ ] Allow for sub-metering (i.e. measuring Phase A and also a washing machine connected to Phase A).
- [ ] Provide other device consumption models.
- [ ] Metered device without state in Home Assistant.
- [ ] Support for batteries.
- [ ] User-friendly UI.
- [ ] Replay of historical data.

