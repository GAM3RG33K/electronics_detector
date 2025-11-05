# Electronics Detection Catalog

**Purpose:** Comprehensive list of electronic devices for energy footprint tracking  
**Last Updated:** November 5, 2025  
**Detection Strategy:** Generic device types only (no brand/SKU identification)

---

## 📱 PERSONAL MOBILE DEVICES

### Currently Detectable (YOLO COCO)
| Device | YOLO Class | Power Range | Priority | Notes |
|--------|-----------|-------------|----------|-------|
| Cell Phone | `cell phone` | 5-10W | ✅ HIGH | Smartphones, basic phones |
| Laptop | `laptop` | 45-65W | ✅ HIGH | All portable computers |

### Expansion Needed (Custom Training)
| Device | Suggested Class | Power Range | Priority | Notes |
|--------|----------------|-------------|----------|-------|
| Tablet | `tablet` | 10-15W | 🔶 MEDIUM | iPads, Android tablets, Surface |
| E-Reader | `e-reader` | 1-3W | 🔶 MEDIUM | Kindle, Kobo, etc. |
| Portable Gaming Console | `handheld console` | 15-30W | 🔵 LOW | Switch, Steam Deck |
| Portable Media Player | `media player` | 2-5W | 🔵 LOW | MP3 players, iPod |

---

## ⌚ WEARABLE ELECTRONICS

### Expansion Needed (Custom Training)
| Device | Suggested Class | Power Range | Priority | Notes |
|--------|----------------|-------------|----------|-------|
| Smartwatch | `smartwatch` | 0.5-2W | 🔶 MEDIUM | Apple Watch, Galaxy Watch, etc. |
| Fitness Tracker | `fitness tracker` | 0.2-1W | 🔶 MEDIUM | Fitbit, activity bands |
| Smart Ring | `smart ring` | 0.1-0.3W | 🔵 LOW | Oura, notification rings |
| Wireless Earbuds | `earbuds` | 1-3W | 🔶 MEDIUM | AirPods, Galaxy Buds (charging case) |
| Wireless Headphones | `headphones` | 2-5W | 🔶 MEDIUM | Over-ear Bluetooth headphones |
| VR Headset | `vr headset` | 20-50W | 🔵 LOW | Quest, Vision Pro |
| AR Glasses | `ar glasses` | 5-15W | 🔵 LOW | Smart glasses |

---

## 💻 COMPUTING & OFFICE

### Currently Detectable (YOLO COCO)
| Device | YOLO Class | Power Range | Priority | Notes |
|--------|-----------|-------------|----------|-------|
| Laptop | `laptop` | 45-65W | ✅ HIGH | Already implemented |
| Keyboard | `keyboard` | 2-5W | ✅ HIGH | Already implemented |
| Mouse | `mouse` | 1-3W | ✅ HIGH | Already implemented |

### Expansion Needed (Custom Training)
| Device | Suggested Class | Power Range | Priority | Notes |
|--------|----------------|-------------|----------|-------|
| Desktop Computer | `desktop` | 100-300W | 🔶 MEDIUM | Tower/all-in-one |
| Monitor | `monitor` | 20-60W | 🔶 MEDIUM | External displays |
| Printer | `printer` | 30-200W | 🔶 MEDIUM | Inkjet, laser, 3D printers |
| Scanner | `scanner` | 10-30W | 🔵 LOW | Document scanners |
| Webcam | `webcam` | 2-5W | 🔵 LOW | External cameras |
| External Hard Drive | `external drive` | 5-10W | 🔵 LOW | HDD/SSD enclosures |
| USB Hub | `usb hub` | 5-15W | 🔵 LOW | Powered hubs |
| Docking Station | `docking station` | 60-100W | 🔵 LOW | Laptop docks |

---

## 🎮 ENTERTAINMENT & GAMING

### Currently Detectable (YOLO COCO)
| Device | YOLO Class | Power Range | Priority | Notes |
|--------|-----------|-------------|----------|-------|
| TV | `tv` | 80-150W | ✅ HIGH | Already implemented |
| Remote | `remote` | 0.5W | ✅ HIGH | Already implemented |

### Expansion Needed (Custom Training)
| Device | Suggested Class | Power Range | Priority | Notes |
|--------|----------------|-------------|----------|-------|
| Gaming Console | `game console` | 100-200W | 🔶 MEDIUM | PlayStation, Xbox, etc. |
| Handheld Gaming | `handheld console` | 15-30W | 🔵 LOW | Switch, Steam Deck |
| Streaming Device | `streaming box` | 5-15W | 🔶 MEDIUM | Roku, Apple TV, Chromecast |
| Speaker (Bluetooth) | `bluetooth speaker` | 5-20W | 🔶 MEDIUM | Portable speakers |
| Smart Speaker | `smart speaker` | 2-6W | 🔶 MEDIUM | Echo, HomePod, Google Home |
| Soundbar | `soundbar` | 30-60W | 🔵 LOW | TV audio systems |
| Projector | `projector` | 150-400W | 🔵 LOW | Home theater projectors |
| DVD/Blu-ray Player | `media player` | 10-25W | 🔵 LOW | Physical media players |

---

## 📷 IMAGING & PHOTOGRAPHY

### Expansion Needed (Custom Training)
| Device | Suggested Class | Power Range | Priority | Notes |
|--------|----------------|-------------|----------|-------|
| Camera (Digital) | `camera` | 5-10W | 🔶 MEDIUM | DSLR, mirrorless (charging) |
| Camcorder | `camcorder` | 8-15W | 🔵 LOW | Video cameras |
| Action Camera | `action camera` | 3-8W | 🔵 LOW | GoPro, similar devices |
| Ring Light | `ring light` | 15-50W | 🔵 LOW | Photography/streaming lights |
| Photo Printer | `photo printer` | 30-80W | 🔵 LOW | Instant photo printers |

---

## 🏠 SMART HOME & IoT

### Expansion Needed (Custom Training)
| Device | Suggested Class | Power Range | Priority | Notes |
|--------|----------------|-------------|----------|-------|
| Smart Display | `smart display` | 10-20W | 🔶 MEDIUM | Echo Show, Nest Hub |
| Security Camera | `security camera` | 3-10W | 🔶 MEDIUM | Indoor/outdoor cams |
| Video Doorbell | `doorbell` | 3-6W | 🔵 LOW | Smart doorbells |
| Smart Thermostat | `thermostat` | 2-5W | 🔵 LOW | Nest, Ecobee |
| Smart Plug | `smart plug` | 1-3W | 🔵 LOW | Power monitoring plugs |
| Smart Light Hub | `light hub` | 2-5W | 🔵 LOW | Philips Hue bridge, etc. |
| Wi-Fi Router | `router` | 5-15W | 🔶 MEDIUM | Wireless routers |
| Mesh Wi-Fi Node | `wifi node` | 5-12W | 🔵 LOW | Mesh network points |
| Network Switch | `network switch` | 5-30W | 🔵 LOW | Ethernet switches |

---

## 🍳 KITCHEN APPLIANCES

### Currently Detectable (YOLO COCO)
| Device | YOLO Class | Power Range | Priority | Notes |
|--------|-----------|-------------|----------|-------|
| Microwave | `microwave` | 600-1200W | ✅ HIGH | Already implemented |
| Oven | `oven` | 2000-5000W | ✅ HIGH | Already implemented |
| Toaster | `toaster` | 800-1500W | ✅ HIGH | Already implemented |
| Refrigerator | `refrigerator` | 100-800W | ✅ HIGH | Already implemented |

### Expansion Needed (Custom Training)
| Device | Suggested Class | Power Range | Priority | Notes |
|--------|----------------|-------------|----------|-------|
| Coffee Maker | `coffee maker` | 600-1200W | 🔶 MEDIUM | Drip, espresso machines |
| Electric Kettle | `kettle` | 1200-2000W | 🔶 MEDIUM | Water kettles |
| Blender | `blender` | 300-1000W | 🔵 LOW | Food blenders |
| Food Processor | `food processor` | 400-800W | 🔵 LOW | Kitchen processors |
| Air Fryer | `air fryer` | 1200-1800W | 🔶 MEDIUM | Increasingly popular |
| Instant Pot | `pressure cooker` | 1000-1200W | 🔵 LOW | Multi-cookers |
| Dishwasher | `dishwasher` | 1200-2400W | 🔶 MEDIUM | Automatic dishwashers |
| Stand Mixer | `mixer` | 250-500W | 🔵 LOW | KitchenAid-style mixers |

---

## 🧹 CLEANING & PERSONAL CARE

### Currently Detectable (YOLO COCO)
| Device | YOLO Class | Power Range | Priority | Notes |
|--------|-----------|-------------|----------|-------|
| Hair Dryer | `hair drier` | 1200-1875W | 🔶 MEDIUM | Available in COCO |

### Expansion Needed (Custom Training)
| Device | Suggested Class | Power Range | Priority | Notes |
|--------|----------------|-------------|----------|-------|
| Vacuum Cleaner | `vacuum` | 500-1500W | 🔶 MEDIUM | Upright, canister, robot |
| Electric Toothbrush | `electric toothbrush` | 1-3W | 🔵 LOW | Charging base |
| Electric Shaver | `electric shaver` | 5-15W | 🔵 LOW | Charging base |
| Curling/Flat Iron | `hair iron` | 25-150W | 🔵 LOW | Hair styling tools |
| Humidifier | `humidifier` | 20-50W | 🔵 LOW | Air moisture devices |
| Air Purifier | `air purifier` | 30-100W | 🔶 MEDIUM | HEPA filters |
| Fan | `fan` | 20-75W | 🔶 MEDIUM | Desk, floor, tower fans |
| Space Heater | `heater` | 750-1500W | 🔶 MEDIUM | Portable heaters |

---

## 🔌 POWER & CHARGING

### Expansion Needed (Custom Training)
| Device | Suggested Class | Power Range | Priority | Notes |
|--------|----------------|-------------|----------|-------|
| Phone Charger | `phone charger` | 5-65W | 🔶 MEDIUM | Wall adapters |
| Laptop Charger | `laptop charger` | 45-140W | 🔶 MEDIUM | Power bricks |
| Wireless Charger | `wireless charger` | 5-15W | 🔵 LOW | Qi charging pads |
| Power Bank | `power bank` | 10-30W | 🔵 LOW | Portable batteries (charging) |
| Power Strip | `power strip` | 0W (passthrough) | 🔵 LOW | Non-smart strips |
| UPS | `ups` | 5-50W | 🔵 LOW | Backup power supplies |

---

## 📊 DETECTION PRIORITY SUMMARY

### ✅ ALREADY IMPLEMENTED (10 devices)
- Cell Phone, Laptop, Keyboard, Mouse, Remote
- TV, Microwave, Oven, Toaster, Refrigerator

### 🔶 HIGH PRIORITY ADDITIONS (28 devices)
**Personal & Computing:**
- Tablet, Smartwatch, Fitness Tracker, Wireless Earbuds/Headphones
- Desktop Computer, Monitor, Printer

**Entertainment:**
- Gaming Console, Streaming Device, Bluetooth Speaker, Smart Speaker

**Smart Home:**
- Smart Display, Security Camera, Wi-Fi Router

**Kitchen:**
- Coffee Maker, Electric Kettle, Air Fryer, Dishwasher

**Cleaning:**
- Hair Dryer, Vacuum Cleaner, Air Purifier, Fan, Space Heater

**Charging:**
- Phone Charger, Laptop Charger

### 🔵 MEDIUM/LOW PRIORITY (40+ devices)
- E-readers, gaming handhelds, VR/AR devices
- Cameras, projectors, IoT devices
- Kitchen gadgets, personal care tools
- Network equipment, power accessories

---

## 🎯 IMPLEMENTATION STRATEGY

### Phase 1: Leverage Existing YOLO Classes
Add these from COCO dataset (just need mapping):
- `clock` - Digital/alarm clocks (2-5W)
- `hair drier` - Hair dryers (1200-1875W)

### Phase 2: High-Value Additions (Custom Training)
Focus on most commonly used personal electronics:
1. Tablet
2. Smartwatch
3. Desktop Monitor
4. Gaming Console
5. Smart Speaker
6. Coffee Maker
7. Hair Dryer (if not using COCO)

### Phase 3: Smart Home Expansion
8. Wi-Fi Router
9. Security Camera
10. Smart Display

### Phase 4: Comprehensive Coverage
- Add remaining medium/low priority devices based on usage analytics
- Consider user feedback and common use cases

---

## 📝 NOTES

### Detection Challenges
- **Wearables:** Very small, often worn (may be occluded)
- **Chargers:** Small form factor, similar appearance
- **Smart Speakers:** Can look like regular speakers
- **Router/Hubs:** Often hidden, similar to other boxes

### Power Consumption Considerations
- All power ranges are **typical charging/operating** values
- Actual consumption varies by model, age, settings
- Values are estimates for user awareness, not precise measurement

### Brand Neutrality
- System detects **generic device types only**
- No brand, model, or SKU identification
- Focus on form factor and typical usage patterns

---

**Total Devices Cataloged:** 80+ generic types  
**Currently Implemented:** 10 (13%)  
**Recommended Next Phase:** 28 high-priority additions (35% coverage)

