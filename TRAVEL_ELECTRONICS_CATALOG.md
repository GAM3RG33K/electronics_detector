# Travel Electronics Detection Catalog

**Purpose:** Detect portable electronics that travelers carry in airports/train stations  
**Last Updated:** November 5, 2025  
**Use Case:** Energy footprint tracking for personal travel electronics  
**Scope:** ONLY devices people bring when traveling (carry-on, personal items, worn)

---

## 🎯 SCOPE DEFINITION

### ✅ IN SCOPE - Travel Electronics
- Fits in carry-on luggage or personal bag
- Commonly brought on planes/trains
- Portable and battery-powered or chargeable
- Personal use during travel

### ❌ OUT OF SCOPE - Not Travel Items
- Large appliances (refrigerators, ovens, microwaves, dishwashers)
- Desktop computers (towers)
- Large kitchen appliances
- Home entertainment systems
- Fixed installations

---

## 📱 CORE TRAVEL ELECTRONICS

### Tier 1: Essential Travel Devices (HIGHEST PRIORITY)

| Device | YOLO Status | Power Range | Detection Priority | Travel Prevalence |
|--------|-------------|-------------|-------------------|-------------------|
| **Smartphone** | ✅ Available (`cell phone`) | 5-10W | 🔴 CRITICAL | 99% of travelers |
| **Laptop** | ✅ Available (`laptop`) | 45-65W | 🔴 CRITICAL | 70% of travelers |
| **Tablet** | ⚠️ Need Training | 10-15W | 🔴 CRITICAL | 40% of travelers |
| **Smartwatch** | ⚠️ Need Training | 0.5-2W | 🟡 HIGH | 35% of travelers |
| **Wireless Earbuds** | ⚠️ Need Training | 1-3W | 🟡 HIGH | 60% of travelers |
| **Wireless Headphones** | ⚠️ Need Training | 2-5W | 🟡 HIGH | 45% of travelers |
| **E-Reader** | ⚠️ Need Training | 1-3W | 🟡 HIGH | 25% of travelers |
| **Portable Charger/Power Bank** | ⚠️ Need Training | 10-30W | 🟡 HIGH | 50% of travelers |

**Implementation Status:**
- ✅ **2/8 already working** (Smartphone, Laptop)
- ⚠️ **6/8 need custom training** (Tablet, Smartwatch, Earbuds, Headphones, E-Reader, Power Bank)

---

## 🎮 ENTERTAINMENT & CONTENT

### Tier 2: Common Travel Entertainment (HIGH PRIORITY)

| Device | YOLO Status | Power Range | Detection Priority | Travel Prevalence |
|--------|-------------|-------------|-------------------|-------------------|
| **Portable Gaming Console** | ⚠️ Need Training | 15-30W | 🟡 HIGH | 15% of travelers |
| **Camera (Digital)** | ⚠️ Need Training | 5-10W | 🟡 HIGH | 30% of travelers |
| **Action Camera** | ⚠️ Need Training | 3-8W | 🟢 MEDIUM | 10% of travelers |
| **Portable Bluetooth Speaker** | ⚠️ Need Training | 5-20W | 🟢 MEDIUM | 20% of travelers |
| **Noise-Canceling Headphones** | ⚠️ Need Training | 2-5W | 🟡 HIGH | 30% of travelers |

**Implementation Status:**
- ✅ **0/5 currently working**
- ⚠️ **5/5 need custom training**

---

## 🔌 CHARGING & ACCESSORIES

### Tier 3: Travel Power Accessories (MEDIUM PRIORITY)

| Device | YOLO Status | Power Range | Detection Priority | Travel Prevalence |
|--------|-------------|-------------|-------------------|-------------------|
| **Phone Charger** | ⚠️ Need Training | 5-65W | 🟡 HIGH | 90% of travelers |
| **Laptop Charger** | ⚠️ Need Training | 45-140W | 🟡 HIGH | 60% of travelers |
| **Multi-Port USB Charger** | ⚠️ Need Training | 30-100W | 🟢 MEDIUM | 35% of travelers |
| **Wireless Charging Pad** | ⚠️ Need Training | 5-15W | 🟢 MEDIUM | 15% of travelers |
| **Charging Cable** | ⚠️ Need Training | N/A (passive) | 🔵 LOW | 95% of travelers |
| **Travel Adapter** | ⚠️ Need Training | N/A (passive) | 🔵 LOW | 40% of travelers |

**Implementation Status:**
- ✅ **0/6 currently working**
- ⚠️ **4/6 need custom training** (excluding passive cables/adapters)

---

## 💼 WORK & PRODUCTIVITY

### Tier 4: Business Travel Electronics (MEDIUM PRIORITY)

| Device | YOLO Status | Power Range | Detection Priority | Travel Prevalence |
|--------|-------------|-------------|-------------------|-------------------|
| **Wireless Mouse** | ✅ Available (`mouse`) | 1-3W | 🟢 MEDIUM | 40% of travelers |
| **Wireless Keyboard** | ✅ Available (`keyboard`) | 2-5W | 🟢 MEDIUM | 25% of travelers |
| **Portable Monitor** | ⚠️ Need Training | 15-30W | 🔵 LOW | 5% of travelers |
| **Document Scanner** | ⚠️ Need Training | 10-20W | 🔵 LOW | 3% of travelers |
| **Presentation Remote** | ✅ Available (`remote`) | 0.5W | 🔵 LOW | 10% of travelers |

**Implementation Status:**
- ✅ **3/5 already working** (Mouse, Keyboard, Remote)
- ⚠️ **2/5 need custom training**

---

## 🏥 HEALTH & WEARABLES

### Tier 5: Health & Fitness Devices (MEDIUM PRIORITY)

| Device | YOLO Status | Power Range | Detection Priority | Travel Prevalence |
|--------|-------------|-------------|-------------------|-------------------|
| **Fitness Tracker** | ⚠️ Need Training | 0.2-1W | 🟢 MEDIUM | 30% of travelers |
| **Smart Ring** | ⚠️ Need Training | 0.1-0.3W | 🔵 LOW | 3% of travelers |
| **Electric Toothbrush (travel)** | ⚠️ Need Training | 1-3W | 🟢 MEDIUM | 25% of travelers |
| **Electric Shaver** | ⚠️ Need Training | 5-15W | 🟢 MEDIUM | 20% of travelers |
| **Hair Dryer (travel)** | ✅ Available (`hair drier`) | 800-1200W | 🟢 MEDIUM | 30% of travelers |
| **Hair Straightener/Curler** | ⚠️ Need Training | 25-150W | 🔵 LOW | 15% of travelers |

**Implementation Status:**
- ✅ **1/6 already working** (Hair Dryer)
- ⚠️ **5/6 need custom training**

---

## 📷 SPECIALIZED TRAVEL TECH

### Tier 6: Enthusiast/Specialized Devices (LOW PRIORITY)

| Device | YOLO Status | Power Range | Detection Priority | Travel Prevalence |
|--------|-------------|-------------|-------------------|-------------------|
| **Drone** | ⚠️ Need Training | 50-100W | 🔵 LOW | 5% of travelers |
| **VR Headset** | ⚠️ Need Training | 20-50W | 🔵 LOW | 3% of travelers |
| **Portable Projector** | ⚠️ Need Training | 50-150W | 🔵 LOW | 2% of travelers |
| **Translation Device** | ⚠️ Need Training | 2-5W | 🔵 LOW | 5% of travelers |
| **Portable WiFi Hotspot** | ⚠️ Need Training | 3-8W | 🟢 MEDIUM | 15% of travelers |
| **GPS Device** | ⚠️ Need Training | 2-5W | 🔵 LOW | 5% of travelers |

**Implementation Status:**
- ✅ **0/6 currently working**
- ⚠️ **6/6 need custom training**

---

## 📊 PRIORITY SUMMARY

### ✅ CURRENTLY WORKING (5 devices)
**Coverage: 5/38 travel electronics = 13%**

| Device | Status | Prevalence |
|--------|--------|------------|
| Smartphone | ✅ Working | 99% |
| Laptop | ✅ Working | 70% |
| Wireless Mouse | ✅ Working | 40% |
| Wireless Keyboard | ✅ Working | 25% |
| Presentation Remote | ✅ Working | 10% |

**Note:** We're only detecting 2 of the top 5 most common travel electronics!

---

### 🔴 CRITICAL ADDITIONS (Top 6 Missing Essentials)

These are **must-have** for travel electronics tracking:

| Rank | Device | Prevalence | Power | Why Critical |
|------|--------|------------|-------|--------------|
| 1 | **Tablet** | 40% | 10-15W | Extremely common, high usage |
| 2 | **Wireless Earbuds** | 60% | 1-3W | Most popular travel audio |
| 3 | **Power Bank** | 50% | 10-30W | Essential for modern travel |
| 4 | **Phone Charger** | 90% | 5-65W | Universal necessity |
| 5 | **Wireless Headphones** | 45% | 2-5W | Very common for long trips |
| 6 | **Smartwatch** | 35% | 0.5-2W | Increasingly common |

**Implementation:** These 6 devices would boost coverage from 13% → 42%

---

### 🟡 HIGH VALUE ADDITIONS (Next 8)

| Device | Prevalence | Power | Value |
|--------|------------|-------|-------|
| Laptop Charger | 60% | 45-140W | High power consumption |
| Camera | 30% | 5-10W | Content creators |
| Fitness Tracker | 30% | 0.2-1W | Health-conscious travelers |
| Noise-Canceling Headphones | 30% | 2-5W | Long-haul travelers |
| E-Reader | 25% | 1-3W | Book lovers |
| Electric Toothbrush | 25% | 1-3W | Daily necessity |
| Portable Speaker | 20% | 5-20W | Social travelers |
| Electric Shaver | 20% | 5-15W | Grooming essential |

**Implementation:** Adding these 8 would reach 55% coverage (21/38 devices)

---

## 🎯 RECOMMENDED IMPLEMENTATION ROADMAP

### Phase 1: Quick Win (Current State) ✅
**Status:** COMPLETE
- Smartphone, Laptop, Mouse, Keyboard, Remote
- **Coverage:** 13% of travel electronics
- **Effort:** None (already done)

### Phase 2: Critical Essentials (Next Sprint) 🔴
**Target:** Top 6 missing devices
- Tablet, Wireless Earbuds, Power Bank, Phone Charger, Wireless Headphones, Smartwatch
- **Coverage:** 42% of travel electronics
- **Effort:** Custom model training required
- **ROI:** Highest - covers most common gaps

### Phase 3: High-Value Expansion 🟡
**Target:** Next 8 devices
- Laptop Charger, Camera, Fitness Tracker, Noise-Canceling Headphones, E-Reader, Electric Toothbrush, Speaker, Shaver
- **Coverage:** 55% of travel electronics
- **Effort:** Additional training iterations
- **ROI:** Good - covers specialized but common needs

### Phase 4: Comprehensive Coverage 🟢
**Target:** Remaining 19 devices
- Gaming consoles, specialized tech, low-prevalence items
- **Coverage:** 100% of cataloged travel electronics
- **Effort:** Extensive training
- **ROI:** Diminishing returns

---

## 🛫 TRAVEL CONTEXT CONSIDERATIONS

### Airport Security Implications
**Note:** Some devices must be removed from bags during security screening:
- Laptops and tablets > 10"
- Power banks (must be carry-on only)
- Cameras and large electronics

**Detection Opportunity:** These items are often visible on security trays!

### Charging Stations
**High-visibility locations for device detection:**
- Airport gate charging stations
- Train station waiting areas
- Café/lounge power outlets
- Airline/train seats with USB ports

### Travel Energy Footprint Context
**Key insights:**
- Average traveler carries 3-5 electronic devices
- Combined charging needs: 50-200W per person
- Battery anxiety drives power bank usage
- Business travelers carry more devices

---

## 📈 DETECTION ACCURACY EXPECTATIONS

### Easy to Detect (High Confidence)
- **Laptops:** Large, distinct shape, common
- **Tablets:** Distinct screen, rectangular
- **Headphones:** Recognizable shape when worn/held
- **Power Banks:** Rectangular brick shape

### Moderate Difficulty
- **Smartphones:** Small, but very common
- **Cameras:** Various form factors
- **Portable Speakers:** Can look like other items
- **Chargers:** Small, cable-like

### Challenging to Detect
- **Earbuds:** Very small, often in case
- **Smartwatches:** Worn on wrist, may be occluded
- **Smart Rings:** Tiny, easily missed
- **Cables/Adapters:** Small, non-distinctive

---

## 💡 DETECTION STRATEGY RECOMMENDATIONS

### Priority Detection Zones
1. **Security Checkpoint Trays** - Items laid flat, fully visible
2. **Charging Stations** - Devices plugged in and stationary
3. **Waiting Areas** - People using devices
4. **Luggage Scanning** - Carry-on contents visible

### Camera Placement Suggestions
- **Overhead view:** Better for trays and charging stations
- **Side angle:** Better for worn devices (watches, headphones)
- **Multiple angles:** Combine for best coverage

### User Interaction Model
**Possible workflow:**
1. Traveler places items in security tray
2. Camera detects visible electronics
3. System estimates total power consumption
4. Display shows energy footprint for trip
5. Suggest optimization (e.g., consolidate chargers)

---

## 🔋 POWER CONSUMPTION INSIGHTS

### Typical Travel Scenarios

**Light Traveler (Phone + Earbuds):**
- Total: ~8W charging needs
- Battery duration: 1-2 days
- Charging cycles: 1-2 per trip

**Standard Traveler (Phone + Laptop + Earbuds + Watch):**
- Total: ~60W charging needs
- Battery duration: 6-12 hours active use
- Charging cycles: 2-3 per trip

**Heavy Traveler (Multiple devices + Power Bank):**
- Total: 100-150W charging needs
- Battery duration: Extended by power bank
- Charging cycles: 3-5+ per trip

### Energy Footprint by Trip Length
- **Short flight (< 3 hrs):** 20-50Wh typical usage
- **Medium flight (3-6 hrs):** 50-100Wh typical usage
- **Long-haul flight (6+ hrs):** 100-300Wh typical usage
- **Multi-day train journey:** 200-500Wh typical usage

---

## 📋 DEVICE CATALOG BY FORM FACTOR

### Handheld Devices (Easy to Spot)
- Smartphone, Tablet, E-Reader, Camera, Gaming Console, Power Bank

### Wearable Devices (On Body)
- Smartwatch, Fitness Tracker, Smart Ring, Earbuds (when worn), Headphones (when worn)

### Accessories (Smaller Items)
- Mouse, Keyboard, Remote, Chargers, Cables, Adapters

### Personal Care (Grooming)
- Electric Toothbrush, Electric Shaver, Hair Dryer, Hair Tools

### Audio Devices (Distinctive Shape)
- Earbuds Case, Headphones, Portable Speaker

---

## 🎯 SUCCESS METRICS

### Detection Performance Goals
- **Accuracy:** > 85% for top 10 devices
- **False Positives:** < 10% rate
- **Detection Speed:** < 200ms per frame
- **Coverage:** 50%+ of common travel electronics (Phase 3 target)

### User Value Metrics
- **Energy Awareness:** Help travelers understand their tech footprint
- **Optimization Suggestions:** Consolidate chargers, use power banks efficiently
- **Sustainability:** Encourage responsible electronics usage

---

## 📝 SUMMARY

**Total Travel Electronics Cataloged:** 38 devices  
**Currently Detectable:** 5 devices (13%)  
**Critical Additions Needed:** 6 devices (to reach 42%)  
**Phase 3 Target:** 21 devices (55% coverage)  

**Key Insight:** Focus on the 11 most common devices (Tiers 1-2) to achieve 55% coverage with maximum ROI.

---

**Next Action:** Decide which devices to prioritize for custom model training, starting with the Critical 6.

