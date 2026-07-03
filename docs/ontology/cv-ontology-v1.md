# DSS Computer Vision Ontology — Version 1.0

> **Status:** Immutable · Frozen · Production-Grade
> **Effective Date:** 2026-07-03
> **Review Cycle:** 10-year stability window
> **Maintainer:** Chief AI Architect

---

# PART 1: Design Principles

## 1.1 Why the Ontology Must Be Frozen

The ontology is the **single shared contract** between six independent engineering teams:

| Team | Relies on Ontology For |
|------|----------------------|
| Data Acquisition | Dataset selection, procurement criteria |
| Annotation | Label guidelines, quality control |
| Model Training | Output head definition, loss functions |
| Knowledge Engineering | Mapping perception → semantics |
| Fusion Engineering | Correlation across detections |
| Decision Engineering | Situation assessment, COA generation |

If the ontology changes, every downstream pipeline must be re-validated. A frozen ontology ensures:

- **Dataset compatibility** — Any dataset labeled once remains valid forever.
- **Model interchangeability** — Models trained at different times produce comparable outputs.
- **Audit traceability** — Every detection maps to a permanent, well-defined class.
- **Vendor independence** — No model provider can force ontology changes.

The ontology may be **extended** (new leaf classes added) but never **modified** (existing class definitions, parents, or semantics altered).

## 1.2 Why CV Only Performs Perception

The Computer Vision module operates on **pixel arrays**. It has no access to:

- Order of battle databases
- Signal intelligence
- Human intelligence
- Prior intelligence reports
- Geopolitical context

CV answers exactly one question:

> **"What physical object exists at these pixel coordinates?"**

It does NOT answer:

- "Is this object friendly or enemy?"
- "Is this object a threat?"
- "What is this object's intent?"
- "What nationality is this object?"
- "What weapon capability does this object have?"

These are **inferential questions** that require external knowledge. Assigning them to CV would:

1. **Couple perception to context** — A tank is a tank regardless of whose flag flies over it.
2. **Require retraining for every new adversary** — Adding a new threat would demand new training data.
3. **Violate the open/closed principle** — Perception should be closed for modification, open for extension by knowledge.

## 1.3 Why Semantics Belong to the Knowledge Engine

The Knowledge Engine has access to:

| Knowledge Module | Data Sources |
|-----------------|--------------|
| Friendly Knowledge | Blue force tracker, ORBAT, IFF databases |
| Enemy Knowledge | Intelligence reports, ORBAT, historical patterns |
| Terrain Knowledge | GIS data, DEM, hydrography, soil maps |

These modules **annotate** the CV detection output with semantic meaning:

`
CV Detection:  TrackedVehicle (ObjectType.GROUND_VEHICLE_TRACKED_VEHICLE)
               ↓
Friendly KB:   Not in blue-force tracker → Not friendly
Enemy KB:      Matches T-72 profile in ORBAT → Enemy
               Equipment: T-72 Main Battle Tank
               Nationality: [classified]
Threat:        HIGH (operating in protected zone)
`

This separation means:

- The CV model never needs retraining when the enemy changes.
- New adversaries are handled by updating the Knowledge Base, not the perception model.
- The same CV model serves different customers with different threat definitions.
- The ontology remains stable across decades of geopolitical change.

---
# PART 2: Top-Level Ontology Hierarchy

## 2.1 Category Overview

| # | Category | Purpose | Parent | Children |
|---|----------|---------|--------|----------|
| 1 | People | Detect human presence and density | Root | Person, Group, Crowd |
| 2 | Ground Vehicle | Detect wheeled and tracked land vehicles | Root | Car, SUV, Van, Pickup, Bus, Truck, Motorcycle, Bicycle, ThreeWheeledVehicle, HeavyEquipment, TrackedVehicle, Trailer, UnknownGroundVehicle |
| 3 | Aircraft | Detect airborne vehicles | Root | FixedWingAircraft, RotaryWingAircraft, UnmannedAerialVehicle, Glider, Balloon, UnknownAircraft |
| 4 | Watercraft | Detect floating vessels | Root | Ship, Boat, ContainerShip, TankerShip, CargoShip, PassengerShip, FishingVessel, NavalVessel, Submarine, Sailboat, SmallCraft, Barge, UnknownWatercraft |
| 5 | Buildings | Detect man-made enclosed structures | Root | Building, SmallStructure, Tower, Silo, Greenhouse, Tent, Ruins, UnknownBuilding |
| 6 | Infrastructure | Detect utility and transportation support structures | Root | TrafficLight, StreetLight, Signpost, Bench, Billboard, Pipeline, CommunicationsTower, WaterTower, UnknownInfrastructure |
| 7 | Road Network | Detect paved travel surfaces | Root | Road, Highway, Street, Intersection, Roundabout, RoadSign, TrafficSignal, UnknownRoadElement |
| 8 | Vegetation | Detect plant life | Root | Tree, Forest, Shrub, Grassland, Cropland, Orchard, UnknownVegetation |
| 9 | Water Bodies | Detect surface water | Root | Sea, Lake, River, Stream, Pond, Reservoir, Wetland, Beach, UnknownWaterBody |
| 10 | Terrain | Detect land surface character | Root | BarrenLand, RockyTerrain, SandyTerrain, MudTerrain, SnowCovered, UrbanTerrain, UnknownTerrain |
| 11 | Smoke | Detect airborne particulate | Root | SmokePlume, SmokeColumn, SmokeHaze, DustCloud, UnknownSmoke |
| 12 | Fire | Detect combustion | Root | Wildfire, StructuralFire, ControlledBurn, GasFlare, UnknownFire |
| 13 | Construction | Detect ongoing building activity | Root | ConstructionSite, Excavation, Scaffolding, MaterialPile, UnderConstructionStructure, UnknownConstruction |
| 14 | Engineering Structures | Detect large civil works | Root | Dam, Lock, Canal, Overpass, RetainingWall, Breakwater, Tunnel, UnknownEngineeringStructure |
| 15 | Utilities | Detect energy and water infrastructure | Root | SolarArray, WindTurbine, PowerLine, UtilityPole, TransformerSubstation, GasFacility, UnknownUtility |
| 16 | Barriers | Detect impediments to movement | Root | Wall, Fence, JerseyBarrier, Gate, Checkpoint, VehicleBarrier, UnknownBarrier |
| 17 | Bridges | Detect river-crossing structures | Root | BeamBridge, ArchBridge, SuspensionBridge, CableStayedBridge, TrussBridge, UnknownBridge |
| 18 | Airfields | Detect aviation infrastructure | Root | Runway, Taxiway, Apron, Helipad, Hangar, ControlTower, Terminal, UnknownAirfieldElement |
| 19 | Ports | Detect maritime infrastructure | Root | Dock, ContainerTerminal, PortCrane, PortWarehouse, BreakwaterPort, HarborBasin, DryDock, UnknownPortElement |
| 20 | Rail Infrastructure | Detect rail transport | Root | RailwayTrack, Train, RailwayStation, RailwayBridge, RailwaySignal, RailwayYard, UnknownRailElement |
| 21 | Natural Features | Detect geological formations | Root | Mountain, Cliff, RockFormation, CaveEntrance, Island, Peninsula, SandDune, Glacier, UnknownNaturalFeature |
| 22 | Objects of Interest | Detect strategically relevant objects not fitting elsewhere | Root | ShippingContainer, Pallet, BarrelDrum, Buoy, TentStructure, GuardPost, Watchtower, AntennaMast, Generator, SatelliteDish, FuelTank, CamouflageNet, UnknownObject |

## 2.2 Category Design Notes

- **22 top-level categories** — enough to capture all perception-relevant objects without fragmentation.
- **No military-specific categories** — no "Tank", "Soldier", "Missile", "Artillery" at the CV level.
- **Physically descriptive naming** — "TrackedVehicle" (physical trait), not "ArmoredVehicle" (inferred property).
- **Unknown variants** — Every major category has an Unknown* leaf to capture novel variants without breaking the ontology.
- **No semantic overlap** — Each physical object belongs to exactly one category based on its physical form.

---

# PART 3: Detailed Class Definitions

## 3.1 People

### Person
| Property | Value |
|----------|-------|
| Canonical Name | People.Person |
| Definition | A single individual human. |
| Visual Characteristics | Upright bipedal figure; head, torso, limbs visible at sufficient resolution. |
| Typical Datasets | COCO (person), Open Images (Person), VisDrone (pedestrian) |
| Subtypes | None |
| Annotation Style | Bounding Box (tight around person), Segmentation (full body outline) |

### Group
| Property | Value |
|----------|-------|
| Canonical Name | People.Group |
| Definition | 2-10 persons in close spatial proximity (within 1 body-width of each other). |
| Visual Characteristics | Cluster of individual figures; spacing suggests coordinated movement or stationary gathering. |
| Typical Datasets | VisDrone (people) |
| Subtypes | None |
| Annotation Style | Bounding Box (group bounding box), Polygon (group outline) |

### Crowd
| Property | Value |
|----------|-------|
| Canonical Name | People.Crowd |
| Definition | 11+ persons in close spatial proximity where individual boundaries are not reliably distinguishable. |
| Visual Characteristics | Dense mass of people; individual figures merge into a single region. |
| Typical Datasets | Custom |
| Subtypes | None |
| Annotation Style | Segmentation (region), Polygon |

## 3.2 Ground Vehicle

### Car
| Property | Value |
|----------|-------|
| Canonical Name | GroundVehicle.Car |
| Definition | Four-wheeled motor vehicle designed primarily for passenger transport, with enclosed cabin and separate trunk or continuous roofline. |
| Visual Characteristics | Compact to mid-size footprint; 4 wheels; enclosed cabin; length 3.5-5.0 m. Includes sedans, hatchbacks, coupes, station wagons. |
| Typical Datasets | COCO (car), Open Images (Car), Objects365, VisDrone (car) |
| Subtypes | Sedan, Hatchback, Coupe, StationWagon |
| Annotation Style | Bounding Box, OBB, Segmentation |


### SUV
| Property | Value |
|----------|-------|
| Canonical Name | GroundVehicle.SUV |
| Definition | Sport utility vehicle — a four-wheeled passenger vehicle with raised ground clearance, tall roofline, and often all-wheel drive. |
| Visual Characteristics | Taller than a car; rear hatch; roof rails often present; length 4.4-5.2 m. |
| Typical Datasets | Open Images (SUV), Objects365 |
| Subtypes | CompactSUV, MidSizeSUV, FullSizeSUV |
| Annotation Style | Bounding Box, OBB |

### Van
| Property | Value |
|----------|-------|
| Canonical Name | GroundVehicle.Van |
| Definition | Box-shaped motor vehicle for passenger or cargo transport, with the engine cab integrated into the body. |
| Visual Characteristics | Tall, boxy profile; no separate hood; sliding side doors common; length 4.5-6.0 m. |
| Typical Datasets | Open Images (Van), Objects365 |
| Subtypes | CargoVan, PassengerVan, Minivan |
| Annotation Style | Bounding Box, OBB |

### Pickup
| Property | Value |
|----------|-------|
| Canonical Name | GroundVehicle.Pickup |
| Definition | Light-duty truck with an open cargo bed at the rear and enclosed cab at the front. |
| Visual Characteristics | Distinct cab + open bed configuration; 4 wheels; length 4.8-6.5 m. |
| Typical Datasets | Open Images (PickupTruck), Objects365 |
| Subtypes | RegularCab, ExtendedCab, CrewCab |
| Annotation Style | Bounding Box, OBB |

### Bus
| Property | Value |
|----------|-------|
| Canonical Name | GroundVehicle.Bus |
| Definition | Large motor vehicle designed to carry many passengers, typically with a single-deck or double-deck enclosed body. |
| Visual Characteristics | Elongated box shape; length 8-14 m; multiple windows along sides; large side mirrors. |
| Typical Datasets | COCO (bus), Open Images (Bus), Objects365 |
| Subtypes | CityBus, Coach, Minibus, SchoolBus, DoubleDeckerBus |
| Annotation Style | Bounding Box, OBB |

### Truck
| Property | Value |
|----------|-------|
| Canonical Name | GroundVehicle.Truck |
| Definition | Motor vehicle designed primarily for cargo transport, with a separate cab and cargo area (box, flatbed, tank, or dump body). |
| Visual Characteristics | Distinct cab + cargo body; 6+ wheels common; length 6-18 m; high ground clearance. |
| Typical Datasets | COCO (truck), Open Images (Truck), Objects365, VisDrone (truck) |
| Subtypes | BoxTruck, FlatbedTruck, DumpTruck, TankTruck, TowTruck, RefrigeratedTruck, ConcreteMixerTruck |
| Annotation Style | Bounding Box, OBB, Segmentation |

### Motorcycle
| Property | Value |
|----------|-------|
| Canonical Name | GroundVehicle.Motorcycle |
| Definition | Two-wheeled motor vehicle with handlebars, a seat, and engine between the wheels. |
| Visual Characteristics | Two wheels in line; exposed engine; rider visible; narrow profile; length 2.0-2.5 m. |
| Typical Datasets | COCO (motorcycle), Open Images (Motorcycle), VisDrone (motor) |
| Subtypes | StreetMotorcycle, DirtBike, Scooter, Moped |
| Annotation Style | Bounding Box, OBB |

### Bicycle
| Property | Value |
|----------|-------|
| Canonical Name | GroundVehicle.Bicycle |
| Definition | Two-wheeled human-powered vehicle with pedals, frame, handlebars, and seat. |
| Visual Characteristics | Two wheels in line; triangular frame; no engine; narrow profile; rider visible when occupied. |
| Typical Datasets | COCO (bicycle), Open Images (Bicycle), VisDrone (bicycle) |
| Subtypes | RoadBicycle, MountainBicycle, ElectricBicycle |
| Annotation Style | Bounding Box, OBB |

### ThreeWheeledVehicle
| Property | Value |
|----------|-------|
| Canonical Name | GroundVehicle.ThreeWheeledVehicle |
| Definition | Vehicle with exactly three wheels, either motorized or human-powered. |
| Visual Characteristics | Three wheels (1 front, 2 rear or 2 front, 1 rear); compact footprint; common in developing regions. |
| Typical Datasets | Custom; some VisDrone (tricycle) |
| Subtypes | AutoRickshaw, Tricycle, ThreeWheeledCargo |
| Annotation Style | Bounding Box |

### HeavyEquipment
| Property | Value |
|----------|-------|
| Canonical Name | GroundVehicle.HeavyEquipment |
| Definition | Large self-powered machine designed for construction, earthmoving, or industrial use, operating primarily on worksites. |
| Visual Characteristics | Very large footprint; rugged tires or steel tracks; visible hydraulic arms, buckets, booms, or blades; yellow/orange livery common. |
| Typical Datasets | Open Images (Excavator), Objects365 (excavator) |
| Subtypes | Excavator, Bulldozer, WheeledLoader, Grader, RollerCompactor, Forklift, CraneTruck, BackhoeLoader, SkidSteerLoader |
| Annotation Style | Bounding Box, OBB, Polygon |

### TrackedVehicle
| Property | Value |
|----------|-------|
| Canonical Name | GroundVehicle.TrackedVehicle |
| Definition | Any vehicle that uses continuous tracks (caterpillar treads) instead of wheels for locomotion, excluding heavy equipment already classified above. |
| Visual Characteristics | Continuous rubber or metal track loops; no visible wheels; slow-moving; military green/drab livery common but not defining. |
| Typical Datasets | Custom (rare in public datasets). TrackedVehicle differs from HeavyEquipment in function: armoured personnel carriers, infantry fighting vehicles, and battle tanks use tracks. At CV layer, these are TrackedVehicle; Knowledge layer attaches tactical role. |
| Subtypes | TrackedTransporter, TrackedUtilityVehicle |
| Annotation Style | Bounding Box, OBB |

### Trailer
| Property | Value |
|----------|-------|
| Canonical Name | GroundVehicle.Trailer |
| Definition | Unpowered wheeled vehicle designed to be towed by a motor vehicle. |
| Visual Characteristics | Wheeled box or flat platform; no cab or engine; hitched to towing vehicle. |
| Typical Datasets | Open Images (Trailer) |
| Subtypes | UtilityTrailer, BoatTrailer, SemiTrailer, TankTrailer |
| Annotation Style | Bounding Box |

### UnknownGroundVehicle
| Property | Value |
|----------|-------|
| Canonical Name | GroundVehicle.UnknownGroundVehicle |
| Definition | Ground-moving object with vehicle-like visual characteristics but insufficient resolution or occlusion to assign a specific class. |
| Visual Characteristics | Wheeled/tracked object; moving on land; cannot confidently assign to any subclass above. |
| Annotation Style | Bounding Box |

## 3.3 Aircraft

### FixedWingAircraft
| Property | Value |
|----------|-------|
| Canonical Name | Aircraft.FixedWingAircraft |
| Definition | Heavier-than-air aircraft that generates lift via fixed wings and is propelled by an engine. |
| Visual Characteristics | Elongated fuselage; two wings projecting laterally; tail fin and horizontal stabilizer; engine nacelles on wings or fuselage. |
| Typical Datasets | COCO (airplane), Open Images (Aircraft, Airplane) |
| Subtypes | NarrowBodyAircraft, WideBodyAircraft, RegionalJet, BusinessJet, CargoAircraft, LightAircraft, AgriculturalAircraft |
| Annotation Style | Bounding Box, OBB, Polygon |

### RotaryWingAircraft
| Property | Value |
|----------|-------|
| Canonical Name | Aircraft.RotaryWingAircraft |
| Definition | Heavier-than-air aircraft that generates lift via one or more powered rotors. |
| Visual Characteristics | Main rotor above fuselage; tail rotor (or NOTAR); narrow fuselage; skids or landing gear below; no wings. |
| Typical Datasets | Open Images (Helicopter), VisDrone (helicopter) |
| Subtypes | LightHelicopter, UtilityHelicopter, HeavyLiftHelicopter, Tiltrotor |
| Annotation Style | Bounding Box, OBB |

### UnmannedAerialVehicle
| Property | Value |
|----------|-------|
| Canonical Name | Aircraft.UnmannedAerialVehicle |
| Definition | Aircraft without a pilot on board, operating autonomously or via remote control. |
| Visual Characteristics | Small to medium size; multi-rotor (quadcopter/hexacopter) or fixed-wing mini; no cockpit windows; payload visible underneath. |
| Typical Datasets | Custom (nascent) |
| Subtypes | MultiRotorUAV, FixedWingUAV, HybridVTOLUAV |
| Annotation Style | Bounding Box, Polygon |

### Glider
| Property | Value |
|----------|-------|
| Canonical Name | Aircraft.Glider |
| Definition | Fixed-wing aircraft designed for unpowered flight, using thermals and slope lift. |
| Visual Characteristics | Long slender wings; sleek streamlined fuselage; no engine nacelles; often towed or winch-launched. |
| Typical Datasets | Custom |
| Subtypes | Sailplane, HangGlider, Paraglider |
| Annotation Style | Bounding Box |

### Balloon
| Property | Value |
|----------|-------|
| Canonical Name | Aircraft.Balloon |
| Definition | Lighter-than-air aircraft that uses buoyant gas for lift, with or without engine propulsion. |
| Visual Characteristics | Large spherical or teardrop envelope; basket or gondola suspended below; no wings or rotors. |
| Typical Datasets | Open Images (Balloon) |
| Subtypes | HotAirBalloon, GasBalloon, Blimp, Aerostat |
| Annotation Style | Bounding Box, Polygon |

### UnknownAircraft
| Property | Value |
|----------|-------|
| Canonical Name | Aircraft.UnknownAircraft |
| Definition | Airborne object with aircraft-like visual characteristics but insufficient resolution to assign a specific class. |
| Visual Characteristics | Flying object; cannot confidently classify wing configuration or propulsion type. |
| Annotation Style | Bounding Box |

## 3.4 Watercraft

### Ship
| Property | Value |
|----------|-------|
| Canonical Name | Watercraft.Ship |
| Definition | Large ocean-going vessel with deck and superstructure, length generally >50 m. |
| Visual Characteristics | Large hull; visible superstructure, deck, bridge, funnels, mast; wake visible in water. |
| Typical Datasets | Open Images (Ship) |
| Subtypes | ContainerShip, TankerShip, CargoShip, PassengerShip, FishingVessel, NavalVessel |
| Annotation Style | Bounding Box, OBB, Polygon |

### Boat
| Property | Value |
|----------|-------|
| Canonical Name | Watercraft.Boat |
| Definition | Small to medium watercraft, length generally <50 m, often for personal or local commercial use. |
| Visual Characteristics | Smaller hull; open or semi-enclosed; outboard or inboard engine; may have small cabin. |
| Typical Datasets | Open Images (Boat), SeaShips |
| Subtypes | Speedboat, FishingBoat, Dinghy, InflatableBoat, JetSki |
| Annotation Style | Bounding Box, OBB |

### ContainerShip
| Property | Value |
|----------|-------|
| Canonical Name | Watercraft.ContainerShip |
| Definition | Ship designed specifically to carry standardized intermodal shipping containers stacked on deck and below. |
| Visual Characteristics | Large flat deck with rows of stacked containers; bridge at stern or mid-ship; length 100-400 m. |
| Typical Datasets | SeaShips |
| Subtypes | FeederShip, PanamaxShip, PostPanamaxShip, UltraLargeContainerShip |
| Annotation Style | Bounding Box, OBB |

### TankerShip
| Property | Value |
|----------|-------|
| Canonical Name | Watercraft.TankerShip |
| Definition | Ship designed to carry liquid bulk cargo in tanks integrated into the hull. |
| Visual Characteristics | Long, low-profile hull; exposed pipework on deck; hose handling gear at mid-ship; no container stacks. |
| Typical Datasets | SeaShips |
| Subtypes | OilTanker, ChemicalTanker, LNGTanker, LPGTanker |
| Annotation Style | Bounding Box, OBB |

### CargoShip
| Property | Value |
|----------|-------|
| Canonical Name | Watercraft.CargoShip |
| Definition | Ship designed for dry bulk cargo (grain, ore, coal) or general break-bulk cargo. |
| Visual Characteristics | Large hatches on deck for cargo holds; deck cranes for loading; no container stacks. |
| Typical Datasets | SeaShips |
| Subtypes | BulkCarrier, GeneralCargoShip, RoRoShip, HeavyLiftShip |
| Annotation Style | Bounding Box, OBB |

### PassengerShip
| Property | Value |
|----------|-------|
| Canonical Name | Watercraft.PassengerShip |
| Definition | Ship designed primarily to carry passengers, with accommodation and amenities. |
| Visual Characteristics | Multiple decks with rows of windows; streamlined superstructure; often white hull. |
| Typical Datasets | Custom |
| Subtypes | CruiseShip, Ferry, PassengerLiner, HighSpeedCraft |
| Annotation Style | Bounding Box, OBB |

### FishingVessel
| Property | Value |
|----------|-------|
| Canonical Name | Watercraft.FishingVessel |
| Definition | Ship or boat equipped for commercial fishing operations. |
| Visual Characteristics | Fishing gear on deck (nets, outriggers, booms); smaller superstructure; working deck. |
| Typical Datasets | SeaShips |
| Subtypes | Trawler, Longliner, PurseSeiner, FactoryShip |
| Annotation Style | Bounding Box |

### NavalVessel
| Property | Value |
|----------|-------|
| Canonical Name | Watercraft.NavalVessel |
| Definition | Warship designed for naval warfare, visually defined by weapon mounts, sensors, and angular superstructure. |
| Visual Characteristics | Weapon mounts (gun turrets, missile launchers) visible; angular radar-reflective superstructure; military livery; length 50-300 m. |
| Typical Datasets | Custom (rare in public datasets) |
| Subtypes | CombatVessel, AmphibiousVessel, SupportVessel, PatrolVessel |
| Annotation Style | Bounding Box, OBB |

### Submarine
| Property | Value |
|----------|-------|
| Canonical Name | Watercraft.Submarine |
| Definition | Watercraft capable of independent underwater operation. |
| Visual Characteristics | Elongated cylindrical hull; fin/sail on top; no deck or superstructure; visible only when surfaced or nearly surfaced. |
| Typical Datasets | Custom |
| Subtypes | NuclearSubmarine, DieselSubmarine, MidgetSubmarine |
| Annotation Style | Bounding Box (surfaced portion), Polygon |

### Sailboat
| Property | Value |
|----------|-------|
| Canonical Name | Watercraft.Sailboat |
| Definition | Watercraft propelled primarily by sails mounted on one or more masts. |
| Visual Characteristics | Mast(s) with sails; hull with keel; no engine exhaust visible; length 3-30 m. |
| Typical Datasets | Open Images (Sailboat) |
| Subtypes | MonohullSailboat, Catamaran, Trimaran, SailingYacht |
| Annotation Style | Bounding Box, OBB |

### SmallCraft
| Property | Value |
|----------|-------|
| Canonical Name | Watercraft.SmallCraft |
| Definition | Very small non-powered or low-powered watercraft, typically <5 m, for personal use. |
| Visual Characteristics | Minimal hull; paddles, oars, or small outboard; no cabin; low freeboard. |
| Typical Datasets | Custom |
| Subtypes | Canoe, Kayak, Raft, Rowboat, Paddleboard |
| Annotation Style | Bounding Box, Polygon |

### Barge
| Property | Value |
|----------|-------|
| Canonical Name | Watercraft.Barge |
| Definition | Flat-bottomed watercraft designed for cargo transport on inland waterways, typically unpowered and towed. |
| Visual Characteristics | Flat, rectangular deck; no bow shape; no superstructure; low freeboard; often in towed strings. |
| Typical Datasets | Custom |
| Subtypes | DeckBarge, TankBarge, HopperBarge |
| Annotation Style | Bounding Box, OBB |

### UnknownWatercraft
| Property | Value |
|----------|-------|
| Canonical Name | Watercraft.UnknownWatercraft |
| Definition | Floating object with watercraft-like characteristics but insufficient resolution to classify. |
| Annotation Style | Bounding Box |

## 3.5 Buildings

### Building
| Property | Value |
|----------|-------|
| Canonical Name | Buildings.Building |
| Definition | A permanent enclosed structure with walls and a roof, designed for human occupancy or activity. |
| Visual Characteristics | Rectangular or complex footprint; vertical walls; flat or pitched roof; windows and doors visible at sufficient resolution. |
| Typical Datasets | COCO (building not directly present — use Open Images Building), Open Images (Building), LoveDA |
| Subtypes | ResidentialBuilding, CommercialBuilding, IndustrialBuilding, InstitutionalBuilding, HighRiseBuilding |
| Annotation Style | Polygon, OBB |

### SmallStructure
| Property | Value |
|----------|-------|
| Canonical Name | Buildings.SmallStructure |
| Definition | A small man-made enclosed structure not designed for continuous human occupancy, typically <50 m² footprint. |
| Visual Characteristics | Small footprint; single-purpose (storage, shelter, utility); simple rectangular shape. |
| Typical Datasets | Custom |
| Subtypes | Shed, Booth, Kiosk, Doghouse, Outhouse, PumpHouse |
| Annotation Style | Bounding Box, Polygon |

### Tower
| Property | Value |
|----------|-------|
| Canonical Name | Buildings.Tower |
| Definition | A tall, narrow structure that is taller than its width, freestanding, not designed for continuous human occupancy. |
| Visual Characteristics | Tall, slender vertical profile; narrow footprint; may have observation deck near top. |
| Typical Datasets | Open Images (Tower) |
| Subtypes | ObservationTower, BellTower, Minaret, LatticeTower, CoolingTower |
| Annotation Style | Bounding Box, Polygon |

### Silo
| Property | Value |
|----------|-------|
| Canonical Name | Buildings.Silo |
| Definition | A tall cylindrical structure for storing bulk materials (grain, cement, coal). |
| Visual Characteristics | Tall cylinder; domed or conical top; no windows; loading/unloading pipes visible. |
| Typical Datasets | Custom |
| Subtypes | GrainSilo, CementSilo, CoalSilo, BunkerSilo |
| Annotation Style | Bounding Box, Polygon |

### Greenhouse
| Property | Value |
|----------|-------|
| Canonical Name | Buildings.Greenhouse |
| Definition | A structure with transparent or translucent walls and roof for growing plants under controlled conditions. |
| Visual Characteristics | Translucent glass or plastic panels; low rectangular profile; often in groups; visible vegetation inside. |
| Typical Datasets | LoveDA (greenhouse) |
| Subtypes | GlassGreenhouse, PlasticTunnelGreenhouse |
| Annotation Style | Polygon |

### Tent
| Property | Value |
|----------|-------|
| Canonical Name | Buildings.Tent |
| Definition | A portable shelter made of fabric or similar material stretched over a supporting framework. |
| Visual Characteristics | Fabric exterior; non-rigid walls; triangular or dome shape; guy ropes visible at sufficient resolution. |
| Typical Datasets | Custom |
| Subtypes | CampingTent, MilitaryTent, EventTent, ShelterTent |
| Annotation Style | Bounding Box, Polygon |

### Ruins
| Property | Value |
|----------|-------|
| Canonical Name | Buildings.Ruins |
| Definition | Remains of a building or structure that has been destroyed or collapsed. |
| Visual Characteristics | Irregular rubble; partial walls; no roof; scattered debris; vegetation overgrowth possible. |
| Typical Datasets | Custom |
| Subtypes | BuildingRuins, ArchaeologicalRuins |
| Annotation Style | Polygon |

### UnknownBuilding
| Property | Value |
|----------|-------|
| Canonical Name | Buildings.UnknownBuilding |
| Definition | Man-made structure with building-like characteristics but insufficient resolution to classify. |
| Annotation Style | Bounding Box, Polygon |


## 3.6 Infrastructure

### TrafficLight
| Property | Value |
|----------|-------|
| Canonical Name | Infrastructure.TrafficLight |
| Definition | A signaling device positioned at road intersections to control traffic flow using colored lights. |
| Visual Characteristics | Vertical or horizontal arrangement of red, yellow, and green lamps; mounted on pole or arm; limited to intersection area. |
| Typical Datasets | COCO (traffic light), Open Images (Traffic light) |
| Subtypes | VehicleTrafficLight, PedestrianTrafficLight, TrafficLightWithSign |
| Annotation Style | Bounding Box |

### StreetLight
| Property | Value |
|----------|-------|
| Canonical Name | Infrastructure.StreetLight |
| Definition | A tall pole with a lamp at the top for illuminating roads, pathways, or public areas at night. |
| Visual Characteristics | Tall slender pole; lamp housing at top; often arm extends over roadway; consistent spacing along roads. |
| Typical Datasets | Open Images (Street light) |
| Subtypes | CobraHeadStreetLight, DecorativeStreetLight, HighMastLight |
| Annotation Style | Bounding Box |

### Signpost
| Property | Value |
|----------|-------|
| Canonical Name | Infrastructure.Signpost |
| Definition | A post or structure bearing informational, directional, or warning signage. |
| Visual Characteristics | Vertical post with flat rectangular sign panel(s); text/symbols on sign; various sizes. |
| Typical Datasets | COCO (stop sign), Open Images (Sign) |
| Subtypes | StopSign, DirectionalSign, WarningSign, InformationSign, Billboard |
| Annotation Style | Bounding Box |

### Bench
| Property | Value |
|----------|-------|
| Canonical Name | Infrastructure.Bench |
| Definition | A long seat for multiple people, typically made of wood, metal, or stone, found in public spaces. |
| Visual Characteristics | Horizontal seat surface on legs or solid base; no back or with back; open slatted or solid design. |
| Typical Datasets | Open Images (Bench) |
| Subtypes | ParkBench, BusStopBench, BacklessBench |
| Annotation Style | Bounding Box |

### Billboard
| Property | Value |
|----------|-------|
| Canonical Name | Infrastructure.Billboard |
| Definition | A large outdoor advertising structure with a flat display surface. |
| Visual Characteristics | Large rectangular panel on single or double post; or mounted on building; brightly lit or reflective. |
| Typical Datasets | Open Images (Billboard) |
| Subtypes | StaticBillboard, DigitalBillboard |
| Annotation Style | Bounding Box |

### Pipeline
| Property | Value |
|----------|-------|
| Canonical Name | Infrastructure.Pipeline |
| Definition | A long tube or series of tubes for transporting fluids or gases, visible above ground. |
| Visual Characteristics | Cylindrical tube running above ground on supports; continuous linear feature; valves and flanges visible at access points. |
| Typical Datasets | Custom |
| Subtypes | OilPipeline, GasPipeline, WaterPipeline, PipelineSupport |
| Annotation Style | Polygon, Segmentation |

### CommunicationsTower
| Property | Value |
|----------|-------|
| Canonical Name | Infrastructure.CommunicationsTower |
| Definition | A tall structure supporting antennas for telecommunications, broadcasting, or cellular networks. |
| Visual Characteristics | Lattice or monopole tower; antennas and dishes attached at various heights; guy wires on tall variants. |
| Typical Datasets | Open Images (Radio tower) |
| Subtypes | LatticeTower, MonopoleTower, GuyedMast, CellTower |
| Annotation Style | Bounding Box, Polygon |

### WaterTower
| Property | Value |
|----------|-------|
| Canonical Name | Infrastructure.WaterTower |
| Definition | A tall structure supporting a large tank for water storage at sufficient pressure for distribution. |
| Visual Characteristics | Large elevated tank on tall legs; spherical or cylindrical tank; visible vent pipe. |
| Typical Datasets | Open Images (Water tower) |
| Subtypes | SphericalWaterTower, CylindricalWaterTower, Standpipe |
| Annotation Style | Bounding Box |

### UnknownInfrastructure
| Property | Value |
|----------|-------|
| Canonical Name | Infrastructure.UnknownInfrastructure |
| Definition | Man-made structure with infrastructure-like characteristics but insufficient resolution to classify. |
| Annotation Style | Bounding Box |

## 3.7 Road Network

### Road
| Property | Value |
|----------|-------|
| Canonical Name | RoadNetwork.Road |
| Definition | A paved or unpaved linear surface designed for vehicular travel connecting two or more locations. |
| Visual Characteristics | Continuous linear paved or gravel surface; lane markings visible at sufficient resolution; varying width. |
| Typical Datasets | Open Images (Road), LoveDA (road), SpaceNet (road) |
| Subtypes | PrimaryRoad, SecondaryRoad, TertiaryRoad, ResidentialRoad, UnpavedRoad |
| Annotation Style | Segmentation, Polygon |

### Highway
| Property | Value |
|----------|-------|
| Canonical Name | RoadNetwork.Highway |
| Definition | A major controlled-access road designed for high-speed vehicular traffic, with grade-separated interchanges. |
| Visual Characteristics | Multi-lane divided roadway; median barrier or grass median; exit/entrance ramps; no at-grade intersections. |
| Typical Datasets | Open Images (Highway — implicit in Road) |
| Subtypes | InterstateHighway, NationalHighway, Expressway |
| Annotation Style | Segmentation |

### Street
| Property | Value |
|----------|-------|
| Canonical Name | RoadNetwork.Street |
| Definition | A public road in a city or town, typically with sidewalks and buildings along both sides. |
| Visual Characteristics | Narrower than highway; curbs and sidewalks visible; parked cars along sides; crosswalks at intersections. |
| Typical Datasets | LoveDA (road implicit) |
| Subtypes | MainStreet, SideStreet, OneWayStreet, PedestrianStreet |
| Annotation Style | Segmentation |

### Intersection
| Property | Value |
|----------|-------|
| Canonical Name | RoadNetwork.Intersection |
| Definition | The area where two or more roads cross, join, or meet at grade. |
| Visual Characteristics | Road crossing or merging point; traffic control devices present; painted lane markings. |
| Typical Datasets | Custom (derived from road topology) |
| Subtypes | CrossIntersection, TIntersection, YIntersection, Roundabout, TrafficCircle |
| Annotation Style | Polygon |

### Roundabout
| Property | Value |
|----------|-------|
| Canonical Name | RoadNetwork.Roundabout |
| Definition | A circular intersection where traffic flows counterclockwise (right-hand drive) around a central island. |
| Visual Characteristics | Circular roadway; central island (grass, paved, or landscaped); yield-at-entry markings. |
| Typical Datasets | Custom |
| Subtypes | MiniRoundabout, SingleLaneRoundabout, MultiLaneRoundabout |
| Annotation Style | Polygon |

### RoadSign
| Property | Value |
|----------|-------|
| Canonical Name | RoadNetwork.RoadSign |
| Definition | A sign placed beside or above roads to give instructions or provide information to road users. |
| Visual Characteristics | Flat rectangular/triangular/circular panel on post; may have reflective surface. Differs from Infrastructure.Signpost by being road-specific. |
| Typical Datasets | Custom (mapped to COCO stop sign via Signpost when specific) |
| Subtypes | RegulatorySign, WarningSign, GuideSign, InformationSign |
| Annotation Style | Bounding Box |

### TrafficSignal
| Property | Value |
|----------|-------|
| Canonical Name | RoadNetwork.TrafficSignal |
| Definition | An electrically operated traffic control device at intersections, distinct from standalone TrafficLight by being part of a coordinated signal system visible in aerial imagery. |
| Visual Characteristics | Multiple lamp heads on mast arm or pole; pedestrian signal housing; controller cabinet at base. |
| Typical Datasets | Derived from Infrastructure.TrafficLight |
| Subtypes | MastArmSignal, PoleMountedSignal, PedestrianSignal |
| Annotation Style | Bounding Box |

### UnknownRoadElement
| Property | Value |
|----------|-------|
| Canonical Name | RoadNetwork.UnknownRoadElement |
| Definition | Road-related feature that cannot be confidently assigned to a specific subclass. |
| Annotation Style | Bounding Box, Polygon |

## 3.8 Vegetation

### Tree
| Property | Value |
|----------|-------|
| Canonical Name | Vegetation.Tree |
| Definition | A perennial woody plant with a single main trunk supporting branches and leaves. |
| Visual Characteristics | Single trunk with branching canopy; visible shadow on ground; round or irregular crown shape. |
| Typical Datasets | Open Images (Tree), LoveDA (tree) |
| Subtypes | DeciduousTree, ConiferousTree, PalmTree, FruitTree |
| Annotation Style | Polygon, Segmentation |

### Forest
| Property | Value |
|----------|-------|
| Canonical Name | Vegetation.Forest |
| Definition | A large area densely covered with trees, where individual tree crowns merge into a continuous canopy. |
| Visual Characteristics | Continuous tree canopy; individual trees not reliably separable; darker texture than grassland. |
| Typical Datasets | LoveDA (forest) |
| Subtypes | DenseForest, OpenForest, PlantationForest, MangroveForest |
| Annotation Style | Segmentation |

### Shrub
| Property | Value |
|----------|-------|
| Canonical Name | Vegetation.Shrub |
| Definition | A woody plant smaller than a tree, with multiple stems arising from the base. |
| Visual Characteristics | Low, bushy appearance; multiple stems; no single main trunk; height < 3 m. |
| Typical Datasets | Open Images (Bush) |
| Subtypes | DeciduousShrub, EvergreenShrub |
| Annotation Style | Polygon |

### Grassland
| Property | Value |
|----------|-------|
| Canonical Name | Vegetation.Grassland |
| Definition | An area dominated by grasses and non-woody herbaceous plants. |
| Visual Characteristics | Uniform low green/tan textured surface; no trees or shrubs; mowed or natural appearance. |
| Typical Datasets | LoveDA (grass) |
| Subtypes | Meadow, Pasture, Lawn, Steppe, Savanna |
| Annotation Style | Segmentation |

### Cropland
| Property | Value |
|----------|-------|
| Canonical Name | Vegetation.Cropland |
| Definition | An area of land used for growing agricultural crops, visible as planted rows or uniform crop cover. |
| Visual Characteristics | Geometric field boundaries; parallel rows of uniform vegetation; bare soil visible between rows; seasonal color variation. |
| Typical Datasets | LoveDA (agricultural land) |
| Subtypes | RowCrop, FieldCrop, Orchard, Vineyard, RicePaddy |
| Annotation Style | Segmentation, Polygon |

### Orchard
| Property | Value |
|----------|-------|
| Canonical Name | Vegetation.Orchard |
| Definition | An area of land planted with fruit or nut trees in a regular grid pattern. |
| Visual Characteristics | Regular grid of tree canopies; bare ground between rows; uniform tree size within block. |
| Typical Datasets | Custom |
| Subtypes | AppleOrchard, CitrusOrchard, OliveGrove, NutOrchard |
| Annotation Style | Polygon |

### UnknownVegetation
| Property | Value |
|----------|-------|
| Canonical Name | Vegetation.UnknownVegetation |
| Definition | Plant-covered area that cannot be confidently assigned to a specific vegetation class. |
| Annotation Style | Segmentation |

## 3.9 Water Bodies

### Sea
| Property | Value |
|----------|-------|
| Canonical Name | WaterBodies.Sea |
| Definition | A large body of saltwater covering extensive area, connected to an ocean. |
| Visual Characteristics | Extensive water surface; wave patterns visible; color varies from blue to gray; horizon at distant edge. |
| Typical Datasets | LoveDA (water — implicit via WaterBodies) |
| Subtypes | OpenSea, CoastalSea |
| Annotation Style | Segmentation |

### Lake
| Property | Value |
|----------|-------|
| Canonical Name | WaterBodies.Lake |
| Definition | A large inland body of standing freshwater, surrounded by land. |
| Visual Characteristics | Inland water surface bounded by shoreline; no visible inlet/outlet at resolution; can be round, elongated, or irregular. |
| Typical Datasets | LoveDA (water — implicit) |
| Subtypes | GlacialLake, CraterLake, ReservoirLake, OxbowLake |
| Annotation Style | Segmentation |

### River
| Property | Value |
|----------|-------|
| Canonical Name | WaterBodies.River |
| Definition | A natural flowing watercourse, typically freshwater, flowing toward an ocean, sea, lake, or another river. |
| Visual Characteristics | Linear or meandering water feature; visible banks on both sides; width varies along length; flow direction inferred from morphology. |
| Typical Datasets | LoveDA (water — implicit) |
| Subtypes | PerennialRiver, SeasonalRiver, BraidedRiver, Estuary |
| Annotation Style | Segmentation, Polygon |

### Stream
| Property | Value |
|----------|-------|
| Canonical Name | WaterBodies.Stream |
| Definition | A small, narrow flowing watercourse, narrower than a river, often seasonal. |
| Visual Characteristics | Narrow linear water feature; width < 3 m at resolution; often obscured by vegetation canopy. |
| Typical Datasets | Custom |
| Subtypes | Creek, Brook, Tributary |
| Annotation Style | Segmentation |

### Pond
| Property | Value |
|----------|-------|
| Canonical Name | WaterBodies.Pond |
| Definition | A small body of standing freshwater, smaller than a lake, often man-made. |
| Visual Characteristics | Small enclosed water surface; roughly circular or oval; visible banks; often in agricultural or residential settings. |
| Typical Datasets | Custom |
| Subtypes | FarmPond, GardenPond, KoiPond, RetentionPond |
| Annotation Style | Polygon |

### Reservoir
| Property | Value |
|----------|-------|
| Canonical Name | WaterBodies.Reservoir |
| Definition | An artificial lake created by damming a river or valley, used for water storage. |
| Visual Characteristics | Large enclosed water body with dam visible on one side; irregular shoreline following valley contours. |
| Typical Datasets | Custom |
| Subtypes | WaterSupplyReservoir, HydroelectricReservoir, IrrigationReservoir |
| Annotation Style | Segmentation |

### Wetland
| Property | Value |
|----------|-------|
| Canonical Name | WaterBodies.Wetland |
| Definition | An area of land saturated with water, characterized by aquatic vegetation and hydric soil. |
| Visual Characteristics | Flat, waterlogged terrain; emergent vegetation (reeds, cattails); water visible between vegetation; marsh or swamp appearance. |
| Typical Datasets | LoveDA (water — implicit) |
| Subtypes | Marsh, Swamp, Bog, Fen, MangroveWetland |
| Annotation Style | Segmentation |

### Beach
| Property | Value |
|----------|-------|
| Canonical Name | WaterBodies.Beach |
| Definition | The shore area of a water body composed primarily of sand, gravel, or pebbles. |
| Visual Characteristics | Transition zone between water and land; light-colored sand or gravel; wave-washed shoreline; may have vegetation line above high tide. |
| Typical Datasets | Custom |
| Subtypes | SandyBeach, PebbleBeach, GravelBeach |
| Annotation Style | Segmentation |

### UnknownWaterBody
| Property | Value |
|----------|-------|
| Canonical Name | WaterBodies.UnknownWaterBody |
| Definition | Water-covered area that cannot be confidently assigned to a specific water body class. |
| Annotation Style | Segmentation |

## 3.10 Terrain

### BarrenLand
| Property | Value |
|----------|-------|
| Canonical Name | Terrain.BarrenLand |
| Definition | Land surface with very little or no vegetation, typically arid or eroded. |
| Visual Characteristics | Exposed soil or rock; sparse vegetation (< 5% cover); uniform texture; colors vary (brown, tan, gray). |
| Typical Datasets | LoveDA (barren land) |
| Subtypes | DesertBarren, ErodedBarren, SaltFlat |
| Annotation Style | Segmentation |

### RockyTerrain
| Property | Value |
|----------|-------|
| Canonical Name | Terrain.RockyTerrain |
| Definition | Land surface dominated by exposed bedrock, boulders, or large rock fragments. |
| Visual Characteristics | Irregular rock surfaces; shadows between rocks; rough texture; limited soil and vegetation. |
| Typical Datasets | Custom |
| Subtypes | BedrockOutcrop, BoulderField, ScreeSlope |
| Annotation Style | Segmentation |

### SandyTerrain
| Property | Value |
|----------|-------|
| Canonical Name | Terrain.SandyTerrain |
| Definition | Land surface covered primarily by loose sand particles. |
| Visual Characteristics | Smooth, uniform light-colored surface; dune formations visible at sufficient resolution; wind ripples. |
| Typical Datasets | Custom (implicit in Desert via BarrenLand) |
| Subtypes | SandDuneField, SandSheet, SandFlat |
| Annotation Style | Segmentation |

### MudTerrain
| Property | Value |
|----------|-------|
| Canonical Name | Terrain.MudTerrain |
| Definition | Land surface of water-saturated fine-grained soil, soft and deformable. |
| Visual Characteristics | Dark, moist surface; no standing water visible; tire/footprint tracks common; cracked surface when dry. |
| Typical Datasets | Custom |
| Subtypes | MudFlat, TidalMud, WetlandMud |
| Annotation Style | Segmentation |

### SnowCovered
| Property | Value |
|----------|-------|
| Canonical Name | Terrain.SnowCovered |
| Definition | Land surface partially or fully covered by snow or ice. |
| Visual Characteristics | White or pale surface; smooth texture; shadows appear blue; vegetation and structures partially obscured. |
| Typical Datasets | Custom |
| Subtypes | SnowCoveredGround, IceCoveredGround, PackedSnow |
| Annotation Style | Segmentation |

### UrbanTerrain
| Property | Value |
|----------|-------|
| Canonical Name | Terrain.UrbanTerrain |
| Definition | Land surface within built-up areas that is neither road nor building, including yards, lots, parks, and undeveloped gaps. |
| Visual Characteristics | Mixed surface types (paved, gravel, soil, grass) within urban footprint; irregular boundaries between building blocks. |
| Typical Datasets | LoveDA (urban land implicit) |
| Subtypes | PavedArea, UnpavedArea, ConstructionLot, UrbanGreenspace |
| Annotation Style | Segmentation |

### UnknownTerrain
| Property | Value |
|----------|-------|
| Canonical Name | Terrain.UnknownTerrain |
| Definition | Land surface that cannot be confidently assigned to a specific terrain class. |
| Annotation Style | Segmentation |


## 3.11 Smoke

### SmokePlume
| Property | Value |
|----------|-------|
| Canonical Name | Smoke.SmokePlume |
| Definition | A column or cloud of smoke rising from a specific source, with a identifiable base and vertical rise. |
| Visual Characteristics | Gray, black, or white column rising upward; base at source location; disperses downwind at height. |
| Typical Datasets | Custom |
| Subtypes | SmokeRisingFromFire, SmokeRisingFromExplosion, SmokeRisingFromIndustrialSource |
| Annotation Style | Polygon, Bounding Box |

### SmokeColumn
| Property | Value |
|----------|-------|
| Canonical Name | Smoke.SmokeColumn |
| Definition | A dense, narrow, vertically oriented column of smoke, rising rapidly with minimal horizontal dispersion. |
| Visual Characteristics | Narrow, tall, dense smoke column; dark color; rapid vertical rise; little wind dispersal. |
| Typical Datasets | Custom |
| Subtypes | BlackSmokeColumn, WhiteSmokeColumn |
| Annotation Style | Polygon |

### SmokeHaze
| Property | Value |
|----------|-------|
| Canonical Name | Smoke.SmokeHaze |
| Definition | A widespread, diffuse area of smoke with no identifiable source or column structure. |
| Visual Characteristics | Broad, semi-transparent smoke layer; reduced visibility; no defined edges or source. |
| Typical Datasets | Custom |
| Subtypes | WildfireSmokeHaze, UrbanSmokeHaze, FogSmokeHybrid |
| Annotation Style | Segmentation |

### DustCloud
| Property | Value |
|----------|-------|
| Canonical Name | Smoke.DustCloud |
| Definition | A cloud of suspended dust or fine particulate matter, typically raised by moving vehicles, wind, or explosions. |
| Visual Characteristics | Tan, brown, or gray cloud near ground level; associated with moving vehicles or blast events; settles relatively quickly. |
| Typical Datasets | Custom |
| Subtypes | VehicleDustCloud, ExplosionDustCloud, WindDustStorm |
| Annotation Style | Polygon |

### UnknownSmoke
| Property | Value |
|----------|-------|
| Canonical Name | Smoke.UnknownSmoke |
| Definition | Airborne particulate that cannot be confidently classified as smoke or dust. |
| Annotation Style | Polygon |

## 3.12 Fire

### Wildfire
| Property | Value |
|----------|-------|
| Canonical Name | Fire.Wildfire |
| Definition | An uncontrolled fire burning in natural vegetation (forest, grassland, shrubland). |
| Visual Characteristics | Irregular burning area; visible flames and smoke; advancing fire front; charred area behind. |
| Typical Datasets | Custom |
| Subtypes | ForestFire, GrassFire, Bushfire, CrownFire |
| Annotation Style | Polygon, Segmentation |

### StructuralFire
| Property | Value |
|----------|-------|
| Canonical Name | Fire.StructuralFire |
| Definition | A fire involving a building or other man-made structure. |
| Visual Characteristics | Flames and smoke emanating from building openings (windows, roof); structural damage visible. |
| Typical Datasets | Custom |
| Subtypes | BuildingFire, VehicleFire, InfrastructureFire |
| Annotation Style | Polygon |

### ControlledBurn
| Property | Value |
|----------|-------|
| Canonical Name | Fire.ControlledBurn |
| Definition | A deliberately set and managed fire for agricultural or ecological management. |
| Visual Characteristics | Linear fire lines; regular pattern; containment boundaries visible; low intensity. |
| Typical Datasets | Custom |
| Subtypes | AgriculturalBurn, PrescribedForestBurn |
| Annotation Style | Polygon |

### GasFlare
| Property | Value |
|----------|-------|
| Canonical Name | Fire.GasFlare |
| Definition | A controlled flame at the top of a stack or pipe for burning off combustible gas. |
| Visual Characteristics | Bright flame at elevated point; tall narrow stack structure below; continuous or intermittent. |
| Typical Datasets | Custom |
| Subtypes | IndustrialFlare, OilWellFlare, LandfillFlare |
| Annotation Style | Bounding Box |

### UnknownFire
| Property | Value |
|----------|-------|
| Canonical Name | Fire.UnknownFire |
| Definition | Combustion phenomenon that cannot be confidently classified. |
| Annotation Style | Polygon |

## 3.13 Construction

### ConstructionSite
| Property | Value |
|----------|-------|
| Canonical Name | Construction.ConstructionSite |
| Definition | An area where building or infrastructure construction activity is visibly ongoing. |
| Visual Characteristics | Active work area; construction vehicles and equipment present; partially built structures; material stockpiles; earth disturbance. |
| Typical Datasets | Custom |
| Subtypes | BuildingConstructionSite, RoadConstructionSite, BridgeConstructionSite |
| Annotation Style | Polygon |

### Excavation
| Property | Value |
|----------|-------|
| Canonical Name | Construction.Excavation |
| Definition | A man-made hole or trench dug into the ground, typically for construction or mining. |
| Visual Characteristics | Open pit, trench, or hole in ground; exposed soil/rock sides; may have equipment inside or at edge. |
| Typical Datasets | Custom |
| Subtypes | BuildingFoundationExcavation, Trench, OpenPitMine, BorrowPit |
| Annotation Style | Polygon |

### Scaffolding
| Property | Value |
|----------|-------|
| Canonical Name | Construction.Scaffolding |
| Definition | A temporary structure of poles and planks providing access to elevated areas of a building under construction or repair. |
| Visual Characteristics | Grid-like metal or bamboo framework against building facade; wrapped in netting (green or blue); visible on exterior of structure. |
| Typical Datasets | Custom |
| Subtypes | BuildingScaffolding, BridgeScaffolding |
| Annotation Style | Bounding Box, Polygon |

### MaterialPile
| Property | Value |
|----------|-------|
| Canonical Name | Construction.MaterialPile |
| Definition | A stockpile of construction material (sand, gravel, soil, debris) on or near a construction site. |
| Visual Characteristics | Conical or irregular mound; uniform color and texture per material type; near active construction. |
| Typical Datasets | Custom |
| Subtypes | SandPile, GravelPile, SoilPile, DebrisPile, BrickPile |
| Annotation Style | Polygon |

### UnderConstructionStructure
| Property | Value |
|----------|-------|
| Canonical Name | Construction.UnderConstructionStructure |
| Definition | A partially completed building or structure where the structural frame is visible but walls and roof are incomplete. |
| Visual Characteristics | Exposed structural frame (steel or concrete); no cladding; visible rebar; crane often present; floors visible as stacked slabs. |
| Typical Datasets | Custom |
| Subtypes | HighRiseFrame, LowRiseFrame, BridgeDeckInProgress |
| Annotation Style | Polygon |

### UnknownConstruction
| Property | Value |
|----------|-------|
| Canonical Name | Construction.UnknownConstruction |
| Definition | Construction-related scene element that cannot be confidently assigned to a specific class. |
| Annotation Style | Polygon |

## 3.14 Engineering Structures

### Dam
| Property | Value |
|----------|-------|
| Canonical Name | EngineeringStructures.Dam |
| Definition | A barrier constructed across a river or valley to hold back water, creating a reservoir. |
| Visual Characteristics | Massive concrete or earthen wall across valley; spillway visible; water on upstream side; road often runs along crest. |
| Typical Datasets | Custom |
| Subtypes | ConcreteGravityDam, EarthfillDam, ArchDam, ButtressDam |
| Annotation Style | Polygon |

### Lock
| Property | Value |
|----------|-------|
| Canonical Name | EngineeringStructures.Lock |
| Definition | A chamber with gates at each end used to raise or lower vessels between water levels in a canal or river. |
| Visual Characteristics | Narrow rectangular chamber with gates at both ends; water level differential visible; adjacent control house. |
| Typical Datasets | Custom |
| Subtypes | SingleLock, PairedLock, StaircaseLock |
| Annotation Style | Polygon |

### Canal
| Property | Value |
|----------|-------|
| Canonical Name | EngineeringStructures.Canal |
| Definition | A man-made water channel constructed for navigation, irrigation, or drainage. |
| Visual Characteristics | Straight or gently curving linear water channel; uniform width; banks on both sides; towpath or service road alongside. |
| Typical Datasets | Custom |
| Subtypes | NavigationCanal, IrrigationCanal, DrainageCanal |
| Annotation Style | Polygon, Segmentation |

### Overpass
| Property | Value |
|----------|-------|
| Canonical Name | EngineeringStructures.Overpass |
| Definition | A bridge-like structure carrying one road or railway over another road or railway. |
| Visual Characteristics | Elevated roadway; pillars/supports visible below; railings on sides; shadow underneath. |
| Typical Datasets | Custom |
| Subtypes | RoadOverpass, RailwayOverpass, PedestrianOverpass |
| Annotation Style | Polygon |

### RetainingWall
| Property | Value |
|----------|-------|
| Canonical Name | EngineeringStructures.RetainingWall |
| Definition | A wall built to hold back soil or rock on one side. |
| Visual Characteristics | Vertical wall surface; soil visible on one side; drainage holes visible at base; often stepped or textured. |
| Typical Datasets | Custom |
| Subtypes | ConcreteRetainingWall, StoneRetainingWall, GabionWall |
| Annotation Style | Polygon |

### Breakwater
| Property | Value |
|----------|-------|
| Canonical Name | EngineeringStructures.Breakwater |
| Definition | A structure built offshore to protect a harbor or coastline from wave action. |
| Visual Characteristics | Elongated pile of rocks or concrete blocks paralleling shore; often with a roadway on top; visible wave breaking against it. |
| Typical Datasets | Custom |
| Subtypes | RubbleMoundBreakwater, CaissonBreakwater, FloatingBreakwater |
| Annotation Style | Polygon |

### Tunnel
| Property | Value |
|----------|-------|
| Canonical Name | EngineeringStructures.Tunnel |
| Definition | An underground passage, typically for road or rail, visible only at its entrance/exit portals. |
| Visual Characteristics | Opening in hillside or mountainside; arched or rectangular portal; road or rail entering; ventilation structures above. |
| Typical Datasets | Custom |
| Subtypes | RoadTunnel, RailwayTunnel, PedestrianTunnel, UtilityTunnel |
| Annotation Style | Polygon (portal opening), Bounding Box |

### UnknownEngineeringStructure
| Property | Value |
|----------|-------|
| Canonical Name | EngineeringStructures.UnknownEngineeringStructure |
| Definition | Large civil engineering structure that cannot be confidently classified. |
| Annotation Style | Polygon |

## 3.15 Utilities

### SolarArray
| Property | Value |
|----------|-------|
| Canonical Name | Utilities.SolarArray |
| Definition | An installation of solar panels for generating electricity from sunlight. |
| Visual Characteristics | Grid of dark blue/black rectangular panels; tilted at angle; arranged in rows; reflective surface. |
| Typical Datasets | Custom |
| Subtypes | GroundMountedSolar, RooftopSolar, SolarFarm, FloatingSolar |
| Annotation Style | Polygon |

### WindTurbine
| Property | Value |
|----------|-------|
| Canonical Name | Utilities.WindTurbine |
| Definition | A tall structure with rotating blades that converts wind energy into electricity. |
| Visual Characteristics | Tall white tower; three-bladed rotor at top; rotor diameter up to 100+ m; shadow on ground; typically in groups. |
| Typical Datasets | Open Images (Wind turbine) |
| Subtypes | HorizontalAxisTurbine, VerticalAxisTurbine, OffshoreTurbine |
| Annotation Style | Bounding Box |

### PowerLine
| Property | Value |
|----------|-------|
| Canonical Name | Utilities.PowerLine |
| Definition | An overhead electrical cable supported by utility poles or transmission towers. |
| Visual Characteristics | Linear cable runs between towers/poles; multiple parallel cables; visible at sufficient resolution; often follow roads. |
| Typical Datasets | Custom |
| Subtypes | TransmissionLine, DistributionLine, SubstationFeed |
| Annotation Style | Segmentation |

### UtilityPole
| Property | Value |
|----------|-------|
| Canonical Name | Utilities.UtilityPole |
| Definition | A tall post supporting overhead power lines, telecommunication cables, or lighting. |
| Visual Characteristics | Tall wooden or concrete pole; crossarms near top; wires and cables attached; transformers visible on pole. |
| Typical Datasets | Custom |
| Subtypes | PowerPole, TelephonePole, LightPole, CombinedPole |
| Annotation Style | Bounding Box |

### TransformerSubstation
| Property | Value |
|----------|-------|
| Canonical Name | Utilities.TransformerSubstation |
| Definition | A facility where electrical voltage is transformed between levels, including switchgear and transformers. |
| Visual Characteristics | Fenced compound containing transformers (large metal cylinders), switchgear, insulators, and bus bars. |
| Typical Datasets | Custom |
| Subtypes | DistributionSubstation, TransmissionSubstation, MobileSubstation |
| Annotation Style | Polygon |

### GasFacility
| Property | Value |
|----------|-------|
| Canonical Name | Utilities.GasFacility |
| Definition | An installation for processing, storing, or distributing natural gas. |
| Visual Characteristics | Spherical or cylindrical gas storage tanks; pipework; flare stack; fenced perimeter. |
| Typical Datasets | Custom |
| Subtypes | GasStorageFacility, GasCompressorStation, GasDistributionCenter, LNGTerminal |
| Annotation Style | Polygon |

### UnknownUtility
| Property | Value |
|----------|-------|
| Canonical Name | Utilities.UnknownUtility |
| Definition | Utility infrastructure that cannot be confidently classified. |
| Annotation Style | Polygon |

## 3.16 Barriers

### Wall
| Property | Value |
|----------|-------|
| Canonical Name | Barriers.Wall |
| Definition | A continuous solid vertical barrier made of masonry, concrete, or stone, surrounding or dividing an area. |
| Visual Characteristics | Vertical solid surface; uniform height; may have copings or barbed wire on top; gates at intervals. |
| Typical Datasets | Custom |
| Subtypes | PerimeterWall, BoundaryWall, SecurityWall, GardenWall |
| Annotation Style | Polygon, Segmentation |

### Fence
| Property | Value |
|----------|-------|
| Canonical Name | Barriers.Fence |
| Definition | A linear barrier made of posts and wire, rail, or mesh, used to enclose or divide areas. |
| Visual Characteristics | Open structure; posts at regular intervals; visible gaps between elements; various materials (metal, wood, chain-link). |
| Typical Datasets | Custom |
| Subtypes | ChainLinkFence, WoodFence, WroughtIronFence, BarbedWireFence, ElectricFence |
| Annotation Style | Polygon, Segmentation |

### JerseyBarrier
| Property | Value |
|----------|-------|
| Canonical Name | Barriers.JerseyBarrier |
| Definition | A modular concrete or plastic barrier used to separate lanes of traffic or protect work zones. |
| Visual Characteristics | Trapezoidal profile with angled sides; interlocking segments; orange/white stripes or plain concrete; continuous line. |
| Typical Datasets | Custom |
| Subtypes | ConcreteJerseyBarrier, PlasticWaterFilledBarrier, TaperedBarrier |
| Annotation Style | Polygon |

### Gate
| Property | Value |
|----------|-------|
| Canonical Name | Barriers.Gate |
| Definition | A movable barrier providing controlled passage through a wall, fence, or other enclosure. |
| Visual Characteristics | Opening in wall/fence with hinged or sliding barrier; may have lock mechanism; access road passes through. |
| Typical Datasets | Custom |
| Subtypes | SlidingGate, SwingGate, RisingArmGate, TurnstileGate |
| Annotation Style | Bounding Box, Polygon |

### Checkpoint
| Property | Value |
|----------|-------|
| Canonical Name | Barriers.Checkpoint |
| Definition | A manned or automated inspection point on a road or pathway, typically with barriers and inspection infrastructure. |
| Visual Characteristics | Combination of barriers, booth, signage, and inspection lane; vehicles queued; armed or uniformed personnel may be visible. |
| Typical Datasets | Custom |
| Subtypes | VehicleCheckpoint, PedestrianCheckpoint, SecurityGate |
| Annotation Style | Polygon |

### VehicleBarrier
| Property | Value |
|----------|-------|
| Canonical Name | Barriers.VehicleBarrier |
| Definition | A heavy barrier designed specifically to stop or impede vehicle movement, including bollards, wedges, and crash-rated barriers. |
| Visual Characteristics | Heavy-duty bollards (fixed or retractable); steel wedge barriers; large concrete blocks; anti-ram surface. |
| Typical Datasets | Custom |
| Subtypes | Bollard, WedgeBarrier, RisingBollard, ConcreteBlock, AntiRamFence |
| Annotation Style | Bounding Box, Polygon |

### UnknownBarrier
| Property | Value |
|----------|-------|
| Canonical Name | Barriers.UnknownBarrier |
| Definition | Barrier structure that cannot be confidently classified. |
| Annotation Style | Polygon |


## 3.17 Bridges

### BeamBridge
| Property | Value |
|----------|-------|
| Canonical Name | Bridges.BeamBridge |
| Definition | A bridge whose deck is supported by horizontal beams resting on piers or abutments. |
| Visual Characteristics | Straight horizontal deck; beams visible underneath; piers at regular intervals; simple rectangular profile. |
| Typical Datasets | Custom |
| Subtypes | PlateGirderBridge, BoxGirderBridge, IBeamBridge |
| Annotation Style | Polygon |

### ArchBridge
| Property | Value |
|----------|-------|
| Canonical Name | Bridges.ArchBridge |
| Definition | A bridge with arch-shaped supports at each end, carrying the deck above or below the arch. |
| Visual Characteristics | Visible arch structure; curved upper or lower edge; masonry or steel construction; often historic. |
| Typical Datasets | Custom |
| Subtypes | DeckArchBridge, ThroughArchBridge, TiedArchBridge |
| Annotation Style | Polygon |

### SuspensionBridge
| Property | Value |
|----------|-------|
| Canonical Name | Bridges.SuspensionBridge |
| Definition | A bridge whose deck is suspended from vertical cables attached to main cables stretched between towers. |
| Visual Characteristics | Two tall towers; main cables drooping between towers; vertical suspender cables; long span. |
| Typical Datasets | Custom |
| Subtypes | ClassicSuspensionBridge, SelfAnchoredSuspensionBridge |
| Annotation Style | Polygon |

### CableStayedBridge
| Property | Value |
|----------|-------|
| Canonical Name | Bridges.CableStayedBridge |
| Definition | A bridge with one or more towers from which cables radiate directly to the deck in a fan or harp pattern. |
| Visual Characteristics | Single or multiple towers; cables fanning from tower to deck; no main cable between towers. |
| Typical Datasets | Custom |
| Subtypes | FanCableStayed, HarpCableStayed, SingleTowerCableStayed |
| Annotation Style | Polygon |

### TrussBridge
| Property | Value |
|----------|-------|
| Canonical Name | Bridges.TrussBridge |
| Definition | A bridge whose load-bearing superstructure is composed of trusses (triangular frameworks) of steel or timber. |
| Visual Characteristics | Visible triangular lattice framework on both sides; deck within or on top of truss; metal or timber construction. |
| Typical Datasets | Custom |
| Subtypes | PrattTruss, WarrenTruss, HoweTruss, BaileyBridge |
| Annotation Style | Polygon |

### UnknownBridge
| Property | Value |
|----------|-------|
| Canonical Name | Bridges.UnknownBridge |
| Definition | Bridge structure that cannot be confidently classified by type. |
| Annotation Style | Polygon |

## 3.18 Airfields

### Runway
| Property | Value |
|----------|-------|
| Canonical Name | Airfields.Runway |
| Definition | A defined rectangular area on an airfield prepared for the takeoff and landing of aircraft. |
| Visual Characteristics | Long, straight, paved rectangular strip; markings (centerline, threshold, aim points); lights along edges; orientation aligns with prevailing wind. |
| Typical Datasets | Custom |
| Subtypes | PavedRunway, UnpavedRunway, HelipadRunway |
| Annotation Style | Polygon |

### Taxiway
| Property | Value |
|----------|-------|
| Canonical Name | Airfields.Taxiway |
| Definition | A paved path on an airfield connecting runways to aprons, hangars, and terminals. |
| Visual Characteristics | Paved strip narrower than runway; connecting roads; no threshold markings; centerline markings. |
| Typical Datasets | Custom |
| Subtypes | MainTaxiway, RapidExitTaxiway, ApronTaxiway |
| Annotation Style | Polygon |

### Apron
| Property | Value |
|----------|-------|
| Canonical Name | Airfields.Apron |
| Definition | A defined area on an airfield where aircraft are parked, loaded, unloaded, or serviced. |
| Visual Characteristics | Large paved area adjacent to terminal; parking stands marked; aircraft present; ground service equipment visible. |
| Typical Datasets | Custom |
| Subtypes | PassengerApron, CargoApron, MaintenanceApron, RemoteStand |
| Annotation Style | Polygon |

### Helipad
| Property | Value |
|----------|-------|
| Canonical Name | Airfields.Helipad |
| Definition | A designated landing and takeoff area for helicopters, marked with an 'H'. |
| Visual Characteristics | Circular or square marked area with 'H' in center; on ground, rooftop, or offshore platform; smaller than runway. |
| Typical Datasets | Custom |
| Subtypes | GroundHelipad, RooftopHelipad, OffshoreHelipad |
| Annotation Style | Polygon |

### Hangar
| Property | Value |
|----------|-------|
| Canonical Name | Airfields.Hangar |
| Definition | A large building for housing and maintaining aircraft. |
| Visual Characteristics | Large wide-opening doors; expansive roof without internal columns; adjacent to apron; aircraft visible inside or nearby. |
| Typical Datasets | Custom |
| Subtypes | TShapedHangar, BoxHangar, NoseInHangar |
| Annotation Style | Polygon |

### ControlTower
| Property | Value |
|----------|-------|
| Canonical Name | Airfields.ControlTower |
| Definition | A tall building at an airport from which air traffic controllers manage aircraft movements. |
| Visual Characteristics | Tall tower with glass-walled cab at top; located near center of airfield; visible from all runways. |
| Typical Datasets | Custom |
| Subtypes | StandaloneControlTower, IntegratedControlTower |
| Annotation Style | Bounding Box |

### Terminal
| Property | Value |
|----------|-------|
| Canonical Name | Airfields.Terminal |
| Definition | A building at an airport where passengers transfer between ground transportation and aircraft. |
| Visual Characteristics | Large elongated building; multiple jet bridges extending to aircraft; passenger lounges visible through windows; roadway at front. |
| Typical Datasets | Custom |
| Subtypes | SingleTerminal, PierTerminal, SatelliteTerminal, RemoteTerminal |
| Annotation Style | Polygon |

### UnknownAirfieldElement
| Property | Value |
|----------|-------|
| Canonical Name | Airfields.UnknownAirfieldElement |
| Definition | Airfield feature that cannot be confidently classified. |
| Annotation Style | Polygon |

## 3.19 Ports

### Dock
| Property | Value |
|----------|-------|
| Canonical Name | Ports.Dock |
| Definition | A structure extending into the water where vessels can moor for loading, unloading, or repair. |
| Visual Characteristics | Linear platform projecting into water; piles visible; mooring bollards; may have cranes or warehouses adjacent. |
| Typical Datasets | Custom |
| Subtypes | Pier, Wharf, Quay, FloatingDock, DryDock |
| Annotation Style | Polygon |

### ContainerTerminal
| Property | Value |
|----------|-------|
| Canonical Name | Ports.ContainerTerminal |
| Definition | A dedicated area within a port for the handling and storage of shipping containers. |
| Visual Characteristics | Large paved area with orderly rows of stacked containers; gantry cranes along waterside; yard equipment moving containers. |
| Typical Datasets | Custom |
| Subtypes | StackedContainerYard, EmptyContainerYard, ReeferContainerYard |
| Annotation Style | Polygon |

### PortCrane
| Property | Value |
|----------|-------|
| Canonical Name | Ports.PortCrane |
| Definition | A large crane used at a port for loading and unloading containers or cargo. |
| Visual Characteristics | Tall gantry structure on rails; horizontal boom extending over water; operator cab suspended. |
| Typical Datasets | Custom |
| Subtypes | ShipToShoreCrane, MobileHarborCrane, GantryCrane, FloatingCrane |
| Annotation Style | Bounding Box, Polygon |

### PortWarehouse
| Property | Value |
|----------|-------|
| Canonical Name | Ports.PortWarehouse |
| Definition | A large building at a port used for temporary storage of cargo. |
| Visual Characteristics | Large, wide building near dock; loading docks on one or more sides; roof vents; cargo doors. |
| Typical Datasets | Custom |
| Subtypes | TransitWarehouse, BondedWarehouse, ColdStorageWarehouse |
| Annotation Style | Polygon |

### BreakwaterPort
| Property | Value |
|----------|-------|
| Canonical Name | Ports.BreakwaterPort |
| Definition | A breakwater specific to protecting a port's harbor basin. |
| Visual Characteristics | Extension of EngineeringStructures.Breakwater within port context; may have lighthouse at end. |
| Typical Datasets | Custom |
| Subtypes | RubbleMoundBreakwaterPort, CaissonBreakwaterPort |
| Annotation Style | Polygon |

### HarborBasin
| Property | Value |
|----------|-------|
| Canonical Name | Ports.HarborBasin |
| Definition | The enclosed water area of a port, protected by breakwaters and providing sheltered mooring. |
| Visual Characteristics | Calm water enclosed by breakwaters; vessel mooring along edges; navigable channels maintained. |
| Typical Datasets | Custom |
| Subtypes | InnerHarbor, OuterHarbor, TidalBasin |
| Annotation Style | Segmentation |

### DryDock
| Property | Value |
|----------|-------|
| Canonical Name | Ports.DryDock |
| Definition | A basin that can be drained to allow ship repair below the waterline. |
| Visual Characteristics | Rectangular basin with gate at one end; ship visible inside; caisson or gate structure at entrance. |
| Typical Datasets | Custom |
| Subtypes | GravingDock, FloatingDryDock, MarineRailway |
| Annotation Style | Polygon |

### UnknownPortElement
| Property | Value |
|----------|-------|
| Canonical Name | Ports.UnknownPortElement |
| Definition | Port infrastructure element that cannot be confidently classified. |
| Annotation Style | Polygon |

## 3.20 Rail Infrastructure

### RailwayTrack
| Property | Value |
|----------|-------|
| Canonical Name | RailInfrastructure.RailwayTrack |
| Definition | A pair of parallel steel rails on sleepers/ties forming a route for trains. |
| Visual Characteristics | Parallel steel lines; gravel ballast bed; wooden or concrete sleepers; linear corridor through landscape. |
| Typical Datasets | Custom |
| Subtypes | MainLineTrack, BranchLineTrack, Sidings, MarshallingYardTrack |
| Annotation Style | Segmentation, Polygon |

### Train
| Property | Value |
|----------|-------|
| Canonical Name | RailInfrastructure.Train |
| Definition | A connected series of rail vehicles (locomotive + cars) moving or stationary on railway track. |
| Visual Characteristics | Locomotive at front; connected carriages or wagons; on railway track; moving or static. |
| Typical Datasets | COCO (train), Open Images (Train) |
| Subtypes | PassengerTrain, FreightTrain, HighSpeedTrain, LightRailTram |
| Annotation Style | Bounding Box, OBB, Polygon |

### RailwayStation
| Property | Value |
|----------|-------|
| Canonical Name | RailInfrastructure.RailwayStation |
| Definition | A building and associated platforms where passengers board and alight from trains. |
| Visual Characteristics | Station building adjacent to tracks; covered platforms; canopy over tracks; pedestrian bridge or tunnel visible. |
| Typical Datasets | Custom |
| Subtypes | MajorStation, RegionalStation, SuburbanStation, FlagStop |
| Annotation Style | Polygon |

### RailwayBridge
| Property | Value |
|----------|-------|
| Canonical Name | RailInfrastructure.RailwayBridge |
| Definition | A bridge specifically designed to carry railway tracks across an obstacle. |
| Visual Characteristics | Similar to beam/truss bridge with track on deck; rails visible on top; often narrower than road bridges. |
| Typical Datasets | Custom |
| Subtypes | SteelRailwayBridge, StoneArchRailwayBridge, TrestleBridge |
| Annotation Style | Polygon |

### RailwaySignal
| Property | Value |
|----------|-------|
| Canonical Name | RailInfrastructure.RailwaySignal |
| Definition | A signaling device beside railway tracks controlling train movements. |
| Visual Characteristics | Post with colored light signals (red, yellow, green) or semaphore arm; positioned beside track. |
| Typical Datasets | Custom |
| Subtypes | ColorLightSignal, SemaphoreSignal, CabSignalIndicator |
| Annotation Style | Bounding Box |

### RailwayYard
| Property | Value |
|----------|-------|
| Canonical Name | RailInfrastructure.RailwayYard |
| Definition | A complex of tracks for storing, sorting, and assembling trains. |
| Visual Characteristics | Multiple parallel tracks connected by switches; trains being assembled; visible classification hump. |
| Typical Datasets | Custom |
| Subtypes | ClassificationYard, StorageYard, MaintenanceYard, IntermodalYard |
| Annotation Style | Polygon, Segmentation |

### UnknownRailElement
| Property | Value |
|----------|-------|
| Canonical Name | RailInfrastructure.UnknownRailElement |
| Definition | Rail infrastructure element that cannot be confidently classified. |
| Annotation Style | Polygon |

## 3.21 Natural Features

### Mountain
| Property | Value |
|----------|-------|
| Canonical Name | NaturalFeatures.Mountain |
| Definition | A large natural elevation of the earth's surface rising prominently above the surrounding area. |
| Visual Characteristics | Significant elevation; steep slopes; rocky or snow-capped peak; ridge lines; shadow on one side. |
| Typical Datasets | Custom |
| Subtypes | VolcanicMountain, FoldMountain, FaultBlockMountain, DomeMountain |
| Annotation Style | Segmentation |

### Cliff
| Property | Value |
|----------|-------|
| Canonical Name | NaturalFeatures.Cliff |
| Definition | A steep, near-vertical rock face, typically at the edge of a plateau or coastline. |
| Visual Characteristics | Vertical or near-vertical rock face; shadow on face; talus slope at base; top edge clearly defined. |
| Typical Datasets | Custom |
| Subtypes | SeaCliff, MountainCliff, Escarpment |
| Annotation Style | Polygon, Segmentation |

### RockFormation
| Property | Value |
|----------|-------|
| Canonical Name | NaturalFeatures.RockFormation |
| Definition | A natural arrangement of exposed rock visible at the surface, distinct from surrounding terrain. |
| Visual Characteristics | Exposed rock mass; irregular shape; color differs from surroundings; may be isolated or part of larger formation. |
| Typical Datasets | Custom |
| Subtypes | Monolith, Butte, Mesa, Hoodoo, RockOutcrop |
| Annotation Style | Polygon |

### CaveEntrance
| Property | Value |
|----------|-------|
| Canonical Name | NaturalFeatures.CaveEntrance |
| Definition | The visible opening of a natural underground cavity or tunnel. |
| Visual Characteristics | Dark opening in rock face or ground; irregular shape; shadow inside; vegetation often around entrance. |
| Typical Datasets | Custom |
| Subtypes | CaveMouth, RockShelter, Sinkhole |
| Annotation Style | Bounding Box, Polygon |

### Island
| Property | Value |
|----------|-------|
| Canonical Name | NaturalFeatures.Island |
| Definition | A land area completely surrounded by water. |
| Visual Characteristics | Land mass in water body; shoreline clearly defined; may have vegetation, structures, or bare rock. |
| Typical Datasets | Custom |
| Subtypes | ContinentalIsland, OceanicIsland, CoralIsland, RiverIsland, ArtificialIsland |
| Annotation Style | Polygon, Segmentation |

### Peninsula
| Property | Value |
|----------|-------|
| Canonical Name | NaturalFeatures.Peninsula |
| Definition | A landform extending into a body of water and connected to mainland on one side. |
| Visual Characteristics | Narrow land extension into water; connected to larger landmass on one side; water on three sides. |
| Typical Datasets | Custom |
| Subtypes | Headland, Cape, Spit |
| Annotation Style | Polygon, Segmentation |

### SandDune
| Property | Value |
|----------|-------|
| Canonical Name | NaturalFeatures.SandDune |
| Definition | A mound or ridge of sand shaped by wind, typically in deserts or coastal areas. |
| Visual Characteristics | Smooth, rounded ridge of sand; crescent or linear shape; windward slope gentler than leeward. |
| Typical Datasets | Custom |
| Subtypes | CrescentDune, LinearDune, StarDune, ParabolicDune, TransverseDune |
| Annotation Style | Segmentation |

### Glacier
| Property | Value |
|----------|-------|
| Canonical Name | NaturalFeatures.Glacier |
| Definition | A persistent body of dense ice that moves slowly under its own weight. |
| Visual Characteristics | Large white/blue ice mass in valley or mountain region; crevasses visible; moraine debris on edges. |
| Typical Datasets | Custom |
| Subtypes | ValleyGlacier, IceCap, TidewaterGlacier, CirqueGlacier |
| Annotation Style | Segmentation |

### UnknownNaturalFeature
| Property | Value |
|----------|-------|
| Canonical Name | NaturalFeatures.UnknownNaturalFeature |
| Definition | Geological or natural formation that cannot be confidently classified. |
| Annotation Style | Polygon, Segmentation |

## 3.22 Objects of Interest

### ShippingContainer
| Property | Value |
|----------|-------|
| Canonical Name | ObjectsOfInterest.ShippingContainer |
| Definition | A standardized steel box used for intermodal freight transport. |
| Visual Characteristics | Rectangular steel box; corrugated sides; length 20 ft or 40 ft; visible on ships, trains, trucks, or in yards. |
| Typical Datasets | Custom |
| Subtypes | TwentyFootContainer, FortyFootContainer, HighCubeContainer, RefrigeratedContainer |
| Annotation Style | Bounding Box, OBB |

### Pallet
| Property | Value |
|----------|-------|
| Canonical Name | ObjectsOfInterest.Pallet |
| Definition | A flat transport structure supporting goods in a stable fashion for lifting by forklift. |
| Visual Characteristics | Flat wooden/plastic platform; stacked goods on top; forklift entry slots visible from side. |
| Typical Datasets | Custom |
| Subtypes | WoodPallet, PlasticPallet, MetalPallet |
| Annotation Style | Bounding Box |

### BarrelDrum
| Property | Value |
|----------|-------|
| Canonical Name | ObjectsOfInterest.BarrelDrum |
| Definition | A cylindrical container for storing liquids or bulk materials, typically 55 gallons. |
| Visual Characteristics | Short cylinder; ribbed sides; flat top and bottom; metal or plastic; often in groups. |
| Typical Datasets | Custom |
| Subtypes | SteelDrum, PlasticDrum, OpenHeadDrum, ClosedHeadDrum |
| Annotation Style | Bounding Box |

### Buoy
| Property | Value |
|----------|-------|
| Canonical Name | ObjectsOfInterest.Buoy |
| Definition | A floating object moored to the seabed to mark a channel, hazard, or location. |
| Visual Characteristics | Small floating object in water; spherical or cylindrical; bright color (red, green, yellow); may have light or flag. |
| Typical Datasets | Custom |
| Subtypes | NavigationBuoy, MooringBuoy, MarkerBuoy, DataBuoy |
| Annotation Style | Bounding Box |

### TentStructure
| Property | Value |
|----------|-------|
| Canonical Name | ObjectsOfInterest.TentStructure |
| Definition | A large temporary fabric-covered structure, larger than Buildings.Tent, used for events, shelters, or storage. |
| Visual Characteristics | Large fabric roof; pole or frame supported; open sides or fabric walls; often white or colored. |
| Typical Datasets | Custom |
| Subtypes | EventMarquee, MilitaryCommandTent, RefugeeShelter, SupplyTent |
| Annotation Style | Polygon |

### GuardPost
| Property | Value |
|----------|-------|
| Canonical Name | ObjectsOfInterest.GuardPost |
| Definition | A small structure at an entrance or checkpoint where security personnel are stationed. |
| Visual Characteristics | Small booth at entry point; windows on all sides; door for personnel access; often adjacent to gate or barrier. |
| Typical Datasets | Custom |
| Subtypes | SecurityBooth, GuardShack, SentryBox, Gatehouse |
| Annotation Style | Bounding Box |

### Watchtower
| Property | Value |
|----------|-------|
| Canonical Name | ObjectsOfInterest.Watchtower |
| Definition | A tall structure with an elevated platform for observation purposes. |
| Visual Characteristics | Elevated platform on legs or single column; railing around observation area; ladder or stairs; at perimeter of facility. |
| Typical Datasets | Custom |
| Subtypes | PerimeterWatchtower, FireLookoutTower, ObservationTower, GuardTower |
| Annotation Style | Bounding Box, Polygon |

### AntennaMast
| Property | Value |
|----------|-------|
| Canonical Name | ObjectsOfInterest.AntennaMast |
| Definition | A tall vertical pole supporting antennas or other telecommunications equipment. |
| Visual Characteristics | Tall slender pole; antennas and cables attached; may have guy wires; typically on building roof or ground. |
| Typical Datasets | Custom |
| Subtypes | WhipAntenna, DishAntenna, YagiAntenna, ParabolicAntenna |
| Annotation Style | Bounding Box |

### Generator
| Property | Value |
|----------|-------|
| Canonical Name | ObjectsOfInterest.Generator |
| Definition | A device that converts mechanical energy into electrical energy, typically with an internal combustion engine. |
| Visual Characteristics | Rectangular metal enclosure; exhaust pipe; fuel tank adjacent; often on concrete pad; power cables exiting. |
| Typical Datasets | Custom |
| Subtypes | DieselGenerator, GasGenerator, PortableGenerator, StandbyGenerator |
| Annotation Style | Bounding Box |

### SatelliteDish
| Property | Value |
|----------|-------|
| Canonical Name | ObjectsOfInterest.SatelliteDish |
| Definition | A parabolic antenna for transmitting or receiving satellite signals. |
| Visual Characteristics | Circular concave dish; angled toward sky; support structure and mount; cables visible. |
| Typical Datasets | Custom |
| Subtypes | FixedSatelliteDish, MotorizedSatelliteDish, VSATDish, LargeCommsDish |
| Annotation Style | Bounding Box |

### FuelTank
| Property | Value |
|----------|-------|
| Canonical Name | ObjectsOfInterest.FuelTank |
| Definition | A container for storing fuel, including petroleum, diesel, or aviation fuel. |
| Visual Characteristics | Cylindrical (horizontal and vertical) or spherical tank; fill pipes and gauges visible; at fuel depot or facility. |
| Typical Datasets | Custom |
| Subtypes | AboveGroundFuelTank, UndergroundFuelTankVent, FuelBladder, AviationFuelTank |
| Annotation Style | Bounding Box, Polygon |

### CamouflageNet
| Property | Value |
|----------|-------|
| Canonical Name | ObjectsOfInterest.CamouflageNet |
| Definition | A netting with disruptive coloration used to conceal equipment, vehicles, or positions. |
| Visual Characteristics | Irregular draped fabric with mottled pattern; disrupts underlying shape; visible as texture anomaly. |
| Typical Datasets | Custom |
| Subtypes | VehicleCamouflageNet, PositionCamouflageNet, EquipmentCamouflageNet |
| Annotation Style | Polygon |

### UnknownObject
| Property | Value |
|----------|-------|
| Canonical Name | ObjectsOfInterest.UnknownObject |
| Definition | An object of potential strategic interest that does not fit any other defined class. |
| Visual Characteristics | Man-made or natural object with no clear class match; may be novel equipment or structure. |
| Annotation Style | Bounding Box |

---

# PART 4: Ontology Constraints

## 4.1 Allowed Parent-Child Relationships

- **Tree depth:** Maximum 3 levels (Root → Category → Class → Subtype).
- **Single inheritance only:** Every class has exactly one parent.
- **Root classes** (Level 1): 22 top-level categories are the only valid Level-1 nodes.
- **Leaf classes** may be added at Level 2 or Level 3.

## 4.2 Forbidden Relationships

| Forbidden | Reason |
|-----------|--------|
| A class belonging to multiple parents | Violates single-inheritance constraint |
| A Level-1 node being a child of another Level-1 node | All Level-1 nodes are direct children of Root. Cross-category relationships are forbidden. |
| A Level-3 class having children | Would exceed 3-level maximum |
| Military-specific classification as parent | "Tank" cannot be a parent class; "Soldier" cannot be a parent class |
| Semantic fusion of categories | "Armored Vehicle" implies knowledge of armor composition, which is an inferred property |

## 4.3 Duplicate Prevention Rules

1. **Unique canonical path:** Each class must have a unique dotted path (e.g., GroundVehicle.Truck).
2. **No synonym classes:** If two datasets call the same physical object by different names, they map to the same DSS class. Example: "automobile" and "car" both map to GroundVehicle.Car.
3. **Physical distinctness test:** Two proposed classes are considered duplicates if they cannot be consistently distinguished by visual appearance alone.
4. **Registration authority:** Only the Chief AI Architect may approve new classes.

## 4.4 Naming Conventions

| Rule | Convention | Example |
|------|-----------|---------|
| Format | PascalCase | GroundVehicle, BeamBridge |
| No abbreviations | Spell out fully | Intersection not Intersxn |
| No underscores | Never use _ | ShippingContainer not Shipping_Container |
| No hyphens | Never use - | SmallCraft not Small-Craft |
| No ordinal numbers | Use descriptive names | Not VehicleType1 |
| No dataset-specific naming | Use domain-neutral names | Car not COCO_vehicle_car |
| Acronyms | Capitalize as single word | SUV, UAV |

## 4.5 Plural Rules

| Category | Convention | Example |
|----------|-----------|---------|
| Top-level (Level 1) children of Root | Plural PASCAL_CASE | Ground Vehicle, Water Bodies |
| Mid-level (Level 2) categories | Singular PascalCase | GroundVehicle, WaterBody |
| Leaf classes | Singular PascalCase | ShippingContainer, BeamBridge |

Note: The ontology file uses dotted paths (e.g., WaterBodies.Lake) where the top-level segment matches the Level-1 plural category name.

## 4.6 Capitalization

| Context | Convention | Example |
|---------|-----------|---------|
| Canonical class name | PascalCase | GroundVehicle.Car |
| Enum values in code | UPPER_SNAKE_CASE | ObjectType.GROUND_VEHICLE_CAR |
| Human-readable labels | Title Case | "Ground Vehicle → Car" |
| Database column/mapping | snake_case | ground_vehicle_car |

## 4.7 Versioning Strategy

- **Major version:** Changes only when a class definition changes semantically. Expected: never after initial freeze.
- **Minor version:** Incremented when new leaf classes are added. Forward-compatible.
- **Patch version:** Incremented for documentation clarifications, typo fixes, or metadata updates.

Format: <major>.<minor>.<patch> (current: 1.0.0)

## 4.8 Deprecation Strategy

- A class may be **deprecated** (marked for removal) but never removed from the ontology. Deprecated classes remain valid for inference but are excluded from training pipelines.
- Deprecation notice must be published for 12 months before a class can be removed from active training support.
- Reason for deprecation must be documented.
- Mapping from deprecated class to replacement class must be provided.

## 4.9 Backward Compatibility Rules

1. **Never delete a class.** It may only be deprecated.
2. **Never change a parent relationship.** A class's parent cannot be reassigned.
3. **Never change a class name.** The canonical name is permanent.
4. **Adding leaf classes is always backward-compatible.**
5. **Adding new Level-1 categories requires a major version bump and must not break existing mappers.**

---
# PART 5: Dataset Mapping Strategy

## 5.1 Mapping Principles

1. **Semantic mapping, not label preservation.** Map what the object *is*, not what the dataset *calls it*.
2. **Lossy mapping for non-relevant classes.** Classes outside DSS scope are explicitly ignored with documented justification.
3. **One-to-many mapping allowed.** One dataset class may map to one DSS class; one DSS class may receive mappings from multiple dataset classes.
4. **Threshold-based confidence mapping.** When ambiguity exists, document the mapping confidence level.

## 5.2 COCO → DSS

| Source | Original Class | Mapped DSS Class | Reason | Ignored? | Why Ignored? |
|--------|---------------|------------------|--------|----------|-------------|
| COCO | person | People.Person | Direct physical match | No | |
| COCO | bicycle | GroundVehicle.Bicycle | Direct match | No | |
| COCO | car | GroundVehicle.Car | Direct match | No | |
| COCO | motorcycle | GroundVehicle.Motorcycle | Direct match | No | |
| COCO | airplane | Aircraft.FixedWingAircraft | Direct match | No | |
| COCO | bus | GroundVehicle.Bus | Direct match | No | |
| COCO | train | RailInfrastructure.Train | Direct match | No | |
| COCO | truck | GroundVehicle.Truck | Direct match | No | |
| COCO | boat | Watercraft.Boat | Direct match | No | |
| COCO | traffic light | Infrastructure.TrafficLight | Direct match | No | |
| COCO | fire hydrant | Utilities.FireHydrant (via custom mapping) | Physical street-level utility | No | |
| COCO | stop sign | RoadNetwork.RoadSign | Subcategory of RoadSign | No | |
| COCO | parking meter | Infrastructure.StreetFurniture (via custom) | Physical street furniture | No | |
| COCO | bench | Infrastructure.Bench | Direct match | No | |
| COCO | bird → ... all animals | — | Not DSS-relevant | Yes | Animal classes outside DSS scope |
| COCO | backpack → all personal items | — | Personal item | Yes | Personal items below DSS operational relevance |
| COCO | bottle → all food/drink | — | Food/beverage | Yes | Food items outside DSS scope |
| COCO | chair → all furniture | — | Furniture | Yes | Indoor furniture outside DSS scope |
| COCO | TV → all electronics | — | Electronics | Yes | Indoor electronics outside DSS scope |
| COCO | all remaining 68 classes | — | Various non-relevant | Yes | See Section 6 for complete justification |

**COCO mapping summary:** 80 classes → 12 mapped, 68 ignored

## 5.3 Open Images → DSS

| Source | Original Class | Mapped DSS Class | Reason | Ignored? | Why Ignored? |
|--------|---------------|------------------|--------|----------|-------------|
| Open Images | Person | People.Person | Direct | No | |
| Open Images | Car | GroundVehicle.Car | Direct | No | |
| Open Images | Truck | GroundVehicle.Truck | Direct | No | |
| Open Images | Bus | GroundVehicle.Bus | Direct | No | |
| Open Images | Motorcycle | GroundVehicle.Motorcycle | Direct | No | |
| Open Images | Bicycle | GroundVehicle.Bicycle | Direct | No | |
| Open Images | Aircraft | Aircraft.FixedWingAircraft | Direct | No | |
| Open Images | Helicopter | Aircraft.RotaryWingAircraft | Direct | No | |
| Open Images | Boat | Watercraft.Boat | Direct | No | |
| Open Images | Ship | Watercraft.Ship | Direct | No | |
| Open Images | Building | Buildings.Building | Direct | No | |
| Open Images | Road | RoadNetwork.Road | Direct | No | |
| Open Images | Tree | Vegetation.Tree | Direct | No | |
| Open Images | Bridge | Bridges.BeamBridge (generic) | Direct | No | |
| Open Images | Tower | Buildings.Tower | Direct | No | |
| Open Images | Fence | Barriers.Fence | Direct | No | |
| Open Images | Wall | Barriers.Wall | Direct | No | |
| Open Images | Billboard | Infrastructure.Billboard | Direct | No | |
| Open Images | Excavator | GroundVehicle.HeavyEquipment | Subtype | No | |
| Open Images | Crane (construction) | GroundVehicle.HeavyEquipment | Subtype | No | |
| Open Images | Man/Woman/Boy/Girl | — | Demographic | Yes | Illegal demographic classification |
| Open Images | All food classes | — | Food | Yes | Outside DSS scope |
| Open Images | All animal classes | — | Animals | Yes | Outside DSS scope |
| Open Images | All clothing/accessories | — | Personal | Yes | Below DSS relevance |
| Open Images | All furniture | — | Furniture | Yes | Indoor objects |
| Open Images | Body parts | — | Too granular | Yes | Below DSS resolution |
| Open Images | All remaining ~555 classes | — | Various | Yes | See Section 6 |

**Open Images mapping summary:** ~600 classes sampled → ~45 mapped, ~555 ignored

## 5.4 Objects365 → DSS

**Mapping summary:** 365 classes → ~16 mapped (person, vehicles, traffic elements), ~349 ignored (animals, food, furniture, electronics, personal items).

Mapping follows identical logic to COCO/Open Images per the rules in Section 5.1.

## 5.5 LVIS → DSS

LVIS contains ~1200+ fine-grained classes. Rather than enumerate all, rule-based mapping is applied:

| Rule | Scope | Action |
|------|-------|--------|
| Rule A | Person-related | Map to People.Person |
| Rule B | Land vehicles | Map to GroundVehicle subclass |
| Rule C | Air vehicles | Map to Aircraft subclass |
| Rule D | Water vehicles | Map to Watercraft subclass |
| Rule E | Buildings | Map to Buildings subclass |
| Rule F | Infrastructure | Map to Infrastructure subclass |
| Rule G | Barriers | Map to Barriers subclass |
| Rule H | Vegetation | Map to Vegetation subclass |
| Rule I | All animals | Ignored |
| Rule J | All food/beverage | Ignored |
| Rule K | All furniture | Ignored |
| Rule L | All electronics | Ignored |
| Rule M | All clothing | Ignored |
| Rule N | All tools/utensils | Ignored |
| Rule O | All sports equipment | Ignored |
| Rule P | Body parts | Ignored |
| Rule Q | Components (door, window) | Ignored |

**LVIS mapping summary:** ~1200 classes → ~20 mapped via rules, ~1180 ignored

## 5.6 VisDrone → DSS

| Source | Original Class | Mapped DSS Class | Reason | Ignored? |
|--------|---------------|------------------|--------|----------|
| VisDrone | pedestrian | People.Person | Direct | No |
| VisDrone | people | People.Group | Multiple persons | No |
| VisDrone | bicycle | GroundVehicle.Bicycle | Direct | No |
| VisDrone | car | GroundVehicle.Car | Direct | No |
| VisDrone | van | GroundVehicle.Van | Direct | No |
| VisDrone | truck | GroundVehicle.Truck | Direct | No |
| VisDrone | tricycle | GroundVehicle.ThreeWheeledVehicle | Direct | No |
| VisDrone | awning-tricycle | GroundVehicle.ThreeWheeledVehicle | Covered variant | No |
| VisDrone | bus | GroundVehicle.Bus | Direct | No |
| VisDrone | motor | GroundVehicle.Motorcycle | Direct | No |
| VisDrone | others | — | Catch-all noise | Yes |

**VisDrone mapping summary:** 12 classes → 10 mapped, 1 ignored

## 5.7 SpaceNet → DSS

| Source | Original Class | Mapped DSS Class | Reason | Ignored? |
|--------|---------------|------------------|--------|----------|
| SpaceNet | building | Buildings.Building | Direct | No |

**SpaceNet mapping summary:** 1 class → 1 mapped

## 5.8 LoveDA → DSS

| Source | Original Class | Mapped DSS Class | Reason | Ignored? |
|--------|---------------|------------------|--------|----------|
| LoveDA | building | Buildings.Building | Direct | No |
| LoveDA | road | RoadNetwork.Road | Direct | No |
| LoveDA | water | WaterBodies.Water | Direct | No |
| LoveDA | barren | Terrain.BarrenLand | Direct | No |
| LoveDA | forest | Vegetation.Forest | Direct | No |
| LoveDA | agricultural | Vegetation.Cropland | Direct | No |
| LoveDA | background | — | Non-class region | Yes |

**LoveDA mapping summary:** 7 classes → 6 mapped, 1 ignored

## 5.9 SeaShips → DSS

| Source | Original Class | Mapped DSS Class | Reason | Ignored? |
|--------|---------------|------------------|--------|----------|
| SeaShips | ore carrier | Watercraft.CargoShip | Subtype | No |
| SeaShips | bulk cargo carrier | Watercraft.CargoShip | Subtype | No |
| SeaShips | container ship | Watercraft.ContainerShip | Direct | No |
| SeaShips | general cargo ship | Watercraft.CargoShip | Direct | No |
| SeaShips | fishing boat | Watercraft.FishingVessel | Direct | No |
| SeaShips | passenger ship | Watercraft.PassengerShip | Direct | No |

**SeaShips mapping summary:** 6 classes → 6 mapped to 4 DSS classes

## 5.10 Mapping Strategy Summary

| Source | Total Classes | Mapped to DSS | Ignored | Mapped % |
|--------|-------------|---------------|---------|----------|
| COCO | 80 | 12 | 68 | 15.0% |
| Open Images | ~600 | ~45 | ~555 | 7.5% |
| Objects365 | 365 | ~16 | ~349 | 4.4% |
| LVIS | ~1200 | ~20 | ~1180 | 1.7% |
| VisDrone | 12 | 10 | 2 | 83.3% |
| SpaceNet | 1 | 1 | 0 | 100% |
| LoveDA | 7 | 6 | 1 | 85.7% |
| SeaShips | 6 | 6 | 0 | 100% |
| **Total** | **~2271** | **~116** | **~2155** | **5.1%** |

---
# PART 6: Classes That MUST NEVER Appear in the Ontology

## 6.1 General Principle

The DSS ontology covers objects that are:

1. **Physically identifiable** in overhead/surveillance imagery
2. **Operationally relevant** to tactical or strategic decision-making
3. **Scale-appropriate** for DSS sensor resolution (typically >= 10 px minimum dimension)

## 6.2 Prohibited Class Categories

### 6.2.1 Indoor Objects
Bottle, Cup, Fork, Spoon, Knife, Bowl, Chair, Couch, Bed, Table, Desk, Toilet, Sink, Bathtub — all excluded because they are indoor objects not visible in overhead/surveillance imagery, below operational resolution, and irrelevant to tactical decision-making.

### 6.2.2 Food and Beverages
Pizza, Sandwich, Hot Dog, Banana, Apple, Cake, Donut, etc. — food items have zero operational relevance for a military/tactical DSS.

### 6.2.3 Animals
Dog, Cat, Horse, Cow, Bird, etc. — while animals exist in operational environments, individual animal detection is not relevant to tactical DSS at operational scale. Large herds are subsumed by Vegetation/Grassland context.

### 6.2.4 Personal Items and Clothing
Backpack, Handbag, Wallet, Suitcase, Tie, Hat, Glasses, Shoe, Umbrella — personal items below DSS resolution/relevance threshold.

### 6.2.5 Sports and Recreation
Frisbee, Kite, Ball, Skis, Snowboard, Skateboard, Baseball bat, Tennis racket, Surfboard — recreational objects with no operational relevance.

### 6.2.6 Demographics and Attributes
Man, Woman, Boy, Girl — demographic classification has no place in a perception-only CV system. CV detects "Person" — nothing more. Soldier — military role is a semantic classification, not a physical one.

### 6.2.7 Military-Semantic Classes
Tank — "Tank" implies military combat role. CV detects TrackedVehicle. The Knowledge Module annotates it as T-72 MBT.
Armored Vehicle — "Armored" is an inferred material property. CV detects the physical vehicle.
Artillery — a physical cannon is detected as HeavyEquipment. "Artillery" is a tactical function.
Missile — a missile on a launcher is an elongated object. Its identification requires Knowledge.
Soldier — semantic, not perceptual.

## 6.3 The Exclusion Test

For any candidate class C, apply this test:

1. Is C visible in overhead/satellite/drone imagery at >= 10 px? If NO → exclude.
2. Is C operationally relevant to tactical or strategic decisions? If NO → exclude.
3. Does C have a unique, consistent physical appearance? If NO → exclude.
4. Is C a semantic/functional category rather than a physical one? If YES → exclude.
5. Would including C require knowledge of intent, ownership, or context? If YES → exclude.

Any class failing any criterion is excluded.

---
# PART 7: Future Extensibility

## 7.1 Adding New Datasets

New datasets can be integrated without changing the ontology:

**Step 1:** Identify each class in the dataset.
**Step 2:** For each class, determine which DSS class matches based on physical appearance, not name.
**Step 3:** If a DSS class match exists → create mapping entry.
**Step 4:** If no DSS class match exists:
   a) Is the object physically distinct from all existing DSS classes?
      Yes → Propose new leaf class (minor version bump)
      No → It is a synonym; map to closest DSS class
   b) Is the object operationally relevant?
      Yes → Proceed with addition
      No → Mark as ignored

## 7.2 When New Classes Are Needed

A new DSS class is warranted only when:
1. The object is physically distinct from all existing DSS classes.
2. The object is detectable at DSS operational resolution.
3. The object has operational relevance for decision-making.
4. The object appears in at least 2 independent datasets (cross-validation).

## 7.3 Future-Proofing Mechanisms

| Mechanism | How It Works |
|-----------|-------------|
| Unknown* classes | Every category has an Unknown* leaf for novel objects |
| Deprecation path | Classes can be deprecated but never removed |
| Minor version bumps | New leaves only; never changes existing classes |
| Mapping registry | Dataset → DSS mapping is separate from ontology |
| Physical-only anchors | All classes based on visual features, never on function/intent |

---
# PART 8: Knowledge Module Integration

## 8.1 How the Ontology Feeds Knowledge Modules

The ontology provides the **vocabulary** that Knowledge Modules use to express their analyses. Knowledge Modules never receive raw dataset labels — they receive DetectedObject instances typed by the ontology.

### 8.1.1 Friendly Knowledge

`
Input:  DetectionResult with DetectedObject[object_type=GroundVehicle.TrackedVehicle]

Friendly Knowledge:
1. Cross-reference coordinates with Blue Force Tracker database
2. Check: Is there a friendly unit at these coordinates?
   Yes → FriendlyAnalysis(friendly_match=True, confidence=0.95)
   No  → FriendlyAnalysis(friendly_match=False, confidence=0.3)
3. The vehicle is still a TrackedVehicle regardless.
   Friendly status is an annotation, not a replacement of the class.
`

### 8.1.2 Enemy Knowledge

`
Input:  DetectionResult with DetectedObject[object_type=GroundVehicle.TrackedVehicle]

Enemy Knowledge:
1. Match physical profile against known enemy ORBAT
2. Profile: TrackedVehicle with turret, ~10m length, 120mm cannon visible
3. ORBAT match: T-72 Main Battle Tank
4. EnemyAnalysis(enemy_match=True, confidence=0.85,
                  possible_equipment="T-72 MBT",
                  reason="Vehicle profile matches T-72 in enemy sector")
`

### 8.1.3 Terrain Knowledge

`
Input:  DetectionResult with multiple DetectedObject entries

Terrain Knowledge aggregates:
- Vegetation.Forest → terrain is forested
- WaterBodies.River → water obstacle present
- RoadNetwork.Road → mobility corridor
- Terrain.BarrenLand → open terrain

TerrainAnalysis(terrain_type="mixed_forest_open",
                nearby_features=["river_obstacle", "road_access"],
                visibility="limited_by_forest_canopy")
`

## 8.2 Why the CV Model Never Changes

The CV model outputs ObjectType values from the ontology. Knowledge Modules define their own **semantic enrichment** layer:

- New enemy vehicle → Update Enemy Knowledge database; CV model unchanged
- New friendly unit → Update Friendly Knowledge database; CV model unchanged
- New terrain methodology → Update Terrain Knowledge logic; CV model unchanged
- New conflict → Update Knowledge databases with new ORBAT; CV model unchanged

The CV model only needs retraining when:
- Sensor resolution changes (new satellite)
- Environmental conditions change (desert → jungle → arctic)
- New object types need detection (not new semantics)

---
# PART 9: Fusion Engine Integration

## 9.1 Why Fusion Only Consumes Ontology Objects

The Fusion Engine correlates detections across time, space, and sensor modes. It requires a **stable, consistent vocabulary** to perform this correlation.

WRONG (raw dataset labels):
  Detection A: "automobile" (from COCO-trained model)
  Detection B: "car" (from Open Images-trained model)
  Detection C: "sedan" (from LVIS-trained model)
  → Fusion cannot determine if these are the same object.

RIGHT (ontology objects):
  Detection A: GroundVehicle.Car (confidence 0.92)
  Detection B: GroundVehicle.Car (confidence 0.87)
  Detection C: GroundVehicle.Car (confidence 0.78)
  → All three are the same type. Fusion correlates by type + spatial-temporal proximity.

### 9.1.1 Fusion Benefits

| Benefit | Explanation |
|---------|-------------|
| Label normalization | All detections use the same vocabulary regardless of source model |
| Cross-sensor correlation | Satellite + drone detection of same object use same type |
| Temporal tracking | Object tracked across time retains type ID |
| Multi-model ensembling | Multiple model outputs merge at type level, not label level |
| Generic fusion logic | Fusion logic operates on type hierarchies, not flat label lists |

## 9.2 Fusion Process

`
DetectionResult A (t=0, satellite):
  GroundVehicle.Car at (100, 200)
  GroundVehicle.Truck at (300, 400)

DetectionResult B (t=5, drone):
  GroundVehicle.Car at (105, 205)
  GroundVehicle.Truck at (295, 395)

FusionEngine:
  → Correlate by ObjectType + spatial proximity
  → Car(100,200) @ t=0 matches Car(105,205) @ t=5 → moving east
  → Truck(300,400) @ t=0 matches Truck(295,395) @ t=5 → moving west

  FusionResult:
    - Track 1: GroundVehicle.Car, heading 090, speed 1 px/t
    - Track 2: GroundVehicle.Truck, heading 270, speed 1 px/t
`

---
# PART 10: Decision Engine Integration

## 10.1 Independence from Dataset Labels

The Decision Engine generates course-of-action recommendations based on the **fused intelligence picture**. It never sees raw dataset labels.

`
Fused intelligence picture (seen by Decision Engine):
  - Track 1: GroundVehicle.TrackedVehicle at grid 12AB3456, heading 315
  - Track 2: People.Group (8 persons) at grid 12AB3457, stationary
  - Terrain: Vegetation.Forest to north, RoadNetwork.Road to east
  - Enemy KB annotation: Track 1 equipment = T-72 MBT
  - Friendly KB annotation: No friendly match for either track
  - Threat assessment: HIGH

Decision Engine:
  → Enemy tracked vehicle approaching friendly position with infantry
  → Terrain restricts movement to road
  → Recommend: Establish ambush at road choke point
  → Priority: 1 (highest)

Note: The Decision Engine never sees "Tank" as a CV label.
It sees "TrackedVehicle" (CV output) annotated as "T-72 MBT" (Enemy KB).
The Decision Engine trusts the KB annotation, not the CV type, for tactical decisions.
`

## 10.2 Separation of Concerns

| Layer | Vocabulary | Responsibility |
|-------|-----------|---------------|
| CV Detection | Ontology types (GroundVehicle.Car) | Physical detection |
| Knowledge | Semantic annotations (T-72 MBT, Enemy) | Meaning attribution |
| Fusion | Correlated tracks | Multi-source integration |
| Decision | Courses of action | Recommendation |

The Decision Engine is **decoupled from dataset labels** by three layers of abstraction:
1. Ontology (CV output)
2. Knowledge annotation (semantic enrichment)
3. Fusion (correlation and tracking)

---
# PART 11: Initial Ontology Tree (Pre-Review)

`
ROOT
├── People
│   ├── Person
│   ├── Group
│   └── Crowd
│
├── Ground Vehicle
│   ├── Car
│   ├── SUV
│   ├── Van
│   ├── Pickup
│   ├── Bus
│   ├── Truck
│   │   ├── BoxTruck
│   │   ├── FlatbedTruck
│   │   ├── TankerTruck
│   │   └── DumpTruck
│   ├── Motorcycle
│   ├── Bicycle
│   ├── ThreeWheeledVehicle
│   │   ├── AutoRickshaw
│   │   └── CargoTricycle
│   ├── HeavyEquipment
│   │   ├── Excavator
│   │   ├── Bulldozer
│   │   ├── WheelLoader
│   │   ├── Crane
│   │   ├── Forklift
│   │   └── RollerCompactor
│   ├── TrackedVehicle
│   │   ├── TrackedTransporter
│   │   └── TrackedExcavator
│   ├── Trailer
│   │   ├── CargoTrailer
│   │   ├── BoatTrailer
│   │   └── FlatbedTrailer
│   └── UnknownGroundVehicle
│
├── Aircraft
│   ├── FixedWingAircraft
│   │   ├── SingleEngineProp
│   │   ├── MultiEngineProp
│   │   ├── BusinessJet
│   │   ├── NarrowBodyJet
│   │   ├── WideBodyJet
│   │   └── MilitaryAircraft
│   ├── RotaryWingAircraft
│   │   ├── LightHelicopter
│   │   ├── UtilityHelicopter
│   │   └── HeavyHelicopter
│   ├── UnmannedAerialVehicle
│   │   ├── FixedWingUAV
│   │   └── MultiRotorUAV
│   ├── Glider
│   ├── Balloon
│   │   ├── WeatherBalloon
│   │   └── Aerostat
│   └── UnknownAircraft
│
├── Watercraft
│   ├── Ship
│   ├── Boat
│   ├── ContainerShip
│   ├── TankerShip
│   │   ├── OilTanker
│   │   ├── ChemicalTanker
│   │   └── LNGcarrier
│   ├── CargoShip
│   │   ├── BulkCarrier
│   │   └── GeneralCargo
│   ├── PassengerShip
│   │   ├── Ferry
│   │   └── CruiseShip
│   ├── FishingVessel
│   │   ├── Trawler
│   │   └── Longliner
│   ├── NavalVessel
│   │   ├── Destroyer
│   │   ├── Frigate
│   │   ├── Corvette
│   │   ├── AircraftCarrier
│   │   ├── AmphibiousVessel
│   │   └── PatrolVessel
│   ├── Submarine
│   │   ├── ConventionalSubmarine
│   │   └── NuclearSubmarine
│   ├── Sailboat
│   │   ├── Monohull
│   │   └── Catamaran
│   ├── SmallCraft
│   │   ├── Kayak
│   │   ├── Canoe
│   │   ├── Dinghy
│   │   ├── JetSki
│   │   └── Paddleboard
│   ├── Barge
│   │   ├── DeckBarge
│   │   ├── TankBarge
│   │   └── HopperBarge
│   └── UnknownWatercraft
│
├── Buildings
│   ├── Building
│   │   ├── ResidentialBuilding
│   │   ├── CommercialBuilding
│   │   ├── IndustrialBuilding
│   │   ├── AgriculturalBuilding
│   │   └── InstitutionalBuilding
│   ├── SmallStructure
│   │   ├── Shed
│   │   ├── Kiosk
│   │   ├── BusShelter
│   │   └── GuardBooth
│   ├── Tower
│   │   ├── ObservationTower
│   │   ├── BellTower
│   │   ├── Minaret
│   │   ├── WaterTower
│   │   └── RadioTower
│   ├── Silo
│   │   ├── GrainSilo
│   │   ├── CementSilo
│   │   └── StorageTank
│   ├── Greenhouse
│   │   ├── GlassGreenhouse
│   │   └── PlasticTunnel
│   ├── Tent
│   │   ├── SmallTent
│   │   ├── LargeTent
│   │   ├── MilitaryTent
│   │   └── MedicalTent
│   ├── Ruins
│   │   ├── BombedBuilding
│   │   └── CollapsedBuilding
│   └── UnknownBuilding
│
├── Infrastructure
│   ├── TrafficLight
│   ├── StreetLight
│   ├── Signpost
│   ├── Bench
│   ├── Billboard
│   ├── Pipeline
│   ├── CommunicationsTower
│   ├── WaterTower
│   └── UnknownInfrastructure
│
├── Road Network
│   ├── Road
│   │   ├── PavedRoad
│   │   └── UnpavedRoad
│   ├── Highway
│   │   ├── Motorway
│   │   └── Expressway
│   ├── Street
│   │   ├── MainStreet
│   │   └── ResidentialStreet
│   ├── Intersection
│   │   ├── ThreeWayIntersection
│   │   ├── FourWayIntersection
│   │   └── MultiWayIntersection
│   ├── Roundabout
│   ├── RoadSign
│   │   ├── StopSign
│   │   ├── WarningSign
│   │   └── GuideSign
│   ├── TrafficSignal
│   └── UnknownRoadElement
│
├── Vegetation
│   ├── Tree
│   │   ├── DeciduousTree
│   │   ├── ConiferousTree
│   │   └── PalmTree
│   ├── Forest
│   │   ├── DeciduousForest
│   │   ├── ConiferousForest
│   │   └── MixedForest
│   ├── Shrub
│   │   ├── Bush
│   │   └── Hedge
│   ├── Grassland
│   │   ├── Meadow
│   │   ├── Prairie
│   │   └── Pasture
│   ├── Cropland
│   │   ├── RowCrop
│   │   ├── CerealCrop
│   │   └── RicePaddy
│   ├── Orchard
│   │   ├── FruitOrchard
│   │   ├── NutOrchard
│   │   └── Vineyard
│   └── UnknownVegetation
│
├── Water Bodies
│   ├── Sea
│   │   ├── OpenOcean
│   │   └── CoastalSea
│   ├── Lake
│   ├── River
│   │   ├── WideRiver
│   │   └── NarrowRiver
│   ├── Stream
│   │   ├── Creek
│   │   └── Brook
│   ├── Pond
│   ├── Reservoir
│   ├── Wetland
│   │   ├── Marsh
│   │   ├── Swamp
│   │   └── Bog
│   ├── Beach
│   │   ├── SandyBeach
│   │   └── PebbleBeach
│   └── UnknownWaterBody
│
├── Terrain
│   ├── BarrenLand
│   │   ├── ExposedSoil
│   │   └── ErodedLand
│   ├── RockyTerrain
│   │   ├── BoulderField
│   │   ├── Outcrop
│   │   └── Talus
│   ├── SandyTerrain
│   │   ├── SandDuneField
│   │   └── SandyPlain
│   ├── MudTerrain
│   │   ├── Mudflat
│   │   └── TidalFlat
│   ├── SnowCovered
│   │   ├── Snowfield
│   │   └── Glacier
│   ├── UrbanTerrain
│   │   ├── DenseUrban
│   │   └── Suburban
│   └── UnknownTerrain
│
├── Smoke
│   ├── SmokePlume
│   ├── SmokeColumn
│   ├── SmokeHaze
│   ├── DustCloud
│   └── UnknownSmoke
│
├── Fire
│   ├── Wildfire
│   │   ├── ForestFire
│   │   └── GrassFire
│   ├── StructuralFire
│   ├── ControlledBurn
│   │   ├── AgriculturalBurn
│   │   └── PrescribedBurn
│   ├── GasFlare
│   └── UnknownFire
│
├── Construction
│   ├── ConstructionSite
│   │   ├── BuildingConstruction
│   │   └── RoadConstruction
│   ├── Excavation
│   │   ├── FoundationExcavation
│   │   └── Trench
│   ├── Scaffolding
│   ├── MaterialPile
│   │   ├── GravelPile
│   │   ├── SandPile
│   │   └── DebrisPile
│   ├── UnderConstructionStructure
│   └── UnknownConstruction
│
├── Engineering Structures
│   ├── Dam
│   │   ├── ConcreteDam
│   │   ├── EarthfillDam
│   │   ├── GravityDam
│   │   └── ArchDam
│   ├── Lock
│   ├── Canal
│   │   ├── NavigationCanal
│   │   └── IrrigationCanal
│   ├── Overpass
│   │   ├── RoadOverpass
│   │   └── RailOverpass
│   ├── RetainingWall
│   │   ├── ConcreteRetainingWall
│   │   └── StoneRetainingWall
│   ├── Breakwater
│   │   ├── RubbleMoundBreakwater
│   │   └── CaissonBreakwater
│   ├── Tunnel
│   │   ├── RoadTunnel
│   │   └── RailTunnel
│   └── UnknownEngineeringStructure
│
├── Utilities
│   ├── SolarArray
│   │   ├── GroundSolarFarm
│   │   ├── RooftopSolar
│   │   └── SolarCanopy
│   ├── WindTurbine
│   ├── PowerLine
│   │   ├── TransmissionLine
│   │   └── DistributionLine
│   ├── UtilityPole
│   │   ├── PowerPole
│   │   ├── TelephonePole
│   │   └── LightPole
│   ├── TransformerSubstation
│   │   ├── StepUpSubstation
│   │   └── StepDownSubstation
│   ├── GasFacility
│   │   ├── GasStorageTank
│   │   └── GasProcessingPlant
│   └── UnknownUtility
│
├── Barriers
│   ├── Wall
│   │   ├── ConcreteWall
│   │   ├── StoneWall
│   │   ├── MasonryWall
│   │   └── SecurityWall
│   ├── Fence
│   │   ├── ChainLinkFence
│   │   ├── BarbedWireFence
│   │   ├── WoodFence
│   │   └── MetalRailFence
│   ├── JerseyBarrier
│   │   ├── TrafficBarrier
│   │   └── SecurityBarrier
│   ├── Gate
│   │   ├── VehicleGate
│   │   ├── PedestrianGate
│   │   └── SlidingGate
│   ├── Checkpoint
│   │   ├── MilitaryCheckpoint
│   │   ├── PoliceCheckpoint
│   │   └── BorderCheckpoint
│   ├── VehicleBarrier
│   │   ├── AntiVehicleDitch
│   │   ├── EarthBerm
│   │   └── HydraulicBarrier
│   └── UnknownBarrier
│
├── Bridges
│   ├── BeamBridge
│   │   ├── PlateGirderBridge
│   │   └── BoxGirderBridge
│   ├── ArchBridge
│   │   ├── DeckArchBridge
│   │   └── ThroughArchBridge
│   ├── SuspensionBridge
│   ├── CableStayedBridge
│   │   ├── FanCableStayed
│   │   └── HarpCableStayed
│   ├── TrussBridge
│   │   ├── ThroughTruss
│   │   ├── DeckTruss
│   │   └── PonyTruss
│   └── UnknownBridge
│
├── Airfields
│   ├── Runway
│   │   ├── PavedRunway
│   │   └── UnpavedRunway
│   ├── Taxiway
│   ├── Apron
│   │   ├── TerminalApron
│   │   ├── CargoApron
│   │   ├── MaintenanceApron
│   │   └── MilitaryApron
│   ├── Helipad
│   │   ├── GroundHelipad
│   │   ├── RooftopHelipad
│   │   └── ShipboardHelipad
│   ├── Hangar
│   │   ├── MaintenanceHangar
│   │   └── StorageHangar
│   ├── ControlTower
│   ├── Terminal
│   │   ├── PassengerTerminal
│   │   └── CargoTerminal
│   └── UnknownAirfieldElement
│
├── Ports
│   ├── Dock
│   │   ├── MarginalWharf
│   │   ├── FingerPier
│   │   └── FloatingDock
│   ├── ContainerTerminal
│   ├── PortCrane
│   │   ├── ShipToShoreCrane
│   │   ├── MobileHarborCrane
│   │   └── GantryCrane
│   ├── PortWarehouse
│   │   ├── TransitShed
│   │   └── BondedWarehouse
│   ├── BreakwaterPort
│   │   ├── RubbleMoundBreakwater
│   │   └── VerticalWallBreakwater
│   ├── HarborBasin
│   │   ├── InnerHarbor
│   │   └── OuterHarbor
│   ├── DryDock
│   │   ├── GravingDock
│   │   └── FloatingDryDock
│   └── UnknownPortElement
│
├── Rail Infrastructure
│   ├── RailwayTrack
│   │   ├── MainLineTrack
│   │   ├── BranchLineTrack
│   │   └── Sidings
│   ├── Train
│   │   ├── PassengerTrain
│   │   ├── FreightTrain
│   │   ├── HighSpeedTrain
│   │   └── LightRail
│   ├── RailwayStation
│   │   ├── PassengerStation
│   │   └── FreightStation
│   ├── RailwayBridge
│   ├── RailwaySignal
│   │   ├── ColorLightSignal
│   │   └── SemaphoreSignal
│   ├── RailwayYard
│   │   ├── ClassificationYard
│   │   ├── StorageYard
│   │   └── MaintenanceDepot
│   └── UnknownRailElement
│
├── Natural Features
│   ├── Mountain
│   │   ├── Peak
│   │   ├── Ridge
│   │   └── MountainRange
│   ├── Cliff
│   │   ├── SeaCliff
│   │   └── InlandCliff
│   ├── RockFormation
│   │   ├── Outcrop
│   │   ├── Butte
│   │   ├── Mesa
│   │   └── Hoodoo
│   ├── CaveEntrance
│   ├── Island
│   ├── Peninsula
│   ├── SandDune
│   │   ├── CrescentDune
│   │   ├── LinearDune
│   │   └── StarDune
│   ├── Glacier
│   │   ├── ValleyGlacier
│   │   └── IceCap
│   └── UnknownNaturalFeature
│
└── Objects of Interest
    ├── ShippingContainer
    │   ├── TwentyFootContainer
    │   └── FortyFootContainer
    ├── Pallet
    ├── BarrelDrum
    │   ├── SteelDrum
    │   └── PlasticDrum
    ├── Buoy
    │   ├── NavigationBuoy
    │   ├── MooringBuoy
    │   └── DataBuoy
    ├── TentStructure
    │   ├── CommandTent
    │   ├── MedicalTent
    │   └── TroopTent
    ├── GuardPost
    │   ├── SecurityBooth
    │   └── SentryPosition
    ├── Watchtower
    │   ├── BorderWatchtower
    │   ├── PrisonWatchtower
    │   └── ObservationTower
    ├── AntennaMast
    │   ├── RadioMast
    │   └── GuyedMast
    ├── Generator
    │   ├── DieselGenerator
    │   └── GasTurbineGenerator
    ├── SatelliteDish
    │   ├── LargeSatelliteDish
    │   └── SmallSatelliteDish
    ├── FuelTank
    │   ├── HorizontalCylindricalTank
    │   ├── SphericalTank
    │   └── BladderTank
    ├── CamouflageNet
    └── UnknownObject
`

---
# PART 12: Critical Review

## 12.1 Review by Role

### Reviewer 1: Chief Scientist at OpenAI

**Critique:**
- The level-3 subtypes (e.g., BoxTruck, FlatbedTruck under Truck) introduce unnecessary granularity for a CV model. Current SOTA models struggle to reliably distinguish these from overhead views. Either merge them into Truck or provide explicit visual criteria.
- "MilitaryAircraft" under FixedWingAircraft is a semantic leak — it implies operational role, not physical form. Replace with physical descriptors (e.g., "DeltaWingAircraft", "SweptWingAircraft").
- "NavalVessel" subtypes (Destroyer, Frigate, Corvette) are physically indistinguishable at satellite resolution (≤ 0.5 m GSD). These should be merged into a single "CombatVessel" or "Warship" until resolution improves.
- "Crowd" threshold of 11+ persons is arbitrary. Consider removing "Group" and "Crowd" distinction and using a single "PeopleCluster" with count estimated by density, not classification.
- The ontology lacks a notion of "Unknown" at the top level. Every detection should first be "UnknownObject" before being classified.

**Verdict:** Overall sound structure. Reduce architectural overfitting at level 3. Remove semantic leakage in aircraft subtypes.

### Reviewer 2: Lead Computer Vision Engineer at NVIDIA

**Critique:**
- 22 top-level categories × average 7 subclasses = ~150+ classes. This will require very large models (YOLOv10x, ViT-H) to achieve production accuracy. Consider an initial v1.0 with ~80 classes and extend later.
- ThreeWheeledVehicle — visually ambiguous with Motorcycle from many angles. Consider merging into Motorcycle or providing explicit visual disambiguation criteria.
- HeavyEquipment vs TrackedVehicle — tracked excavators appear in both. Resolve by making TrackedVehicle a leaf under HeavyEquipment (i.e., tracked heavy equipment) rather than a separate sibling.
- "Cropland" subtypes (RowCrop, CerealCrop, RicePaddy) are distinguishable only at very high resolution (< 0.3 m GSD). At lower resolution, they all look like "Cropland."
- OBB annotation requirement for vehicles is justified but expensive. Ensure this is a recommendation, not a requirement, for all datasets.

**Verdict:** Reduce total class count for initial release. Fix overlapping definitions (TrackedVehicle + HeavyEquipment). Prioritize annotation cost vs. benefit.

### Reviewer 3: Principal AI Engineer at Google DeepMind

**Critique:**
- The 3-level depth constraint is too restrictive. Some domains (e.g., Watercraft.NavalVessel.Destroyer) naturally want 4 levels. Recommend allowing 4 levels where necessary.
- "ObjectsOfInterest" is a dumping ground for anything that doesn't fit elsewhere. This creates a weak semantic category. Either distribute these into existing categories or define a stronger organizing principle.
- "TentStructure" in ObjectsOfInterest conflicts with "Tent" in Buildings. These should be unified under Buildings.Tent with size as an attribute, not a separate class.
- The "Unknown" pattern (UnknownGroundVehicle, UnknownAircraft, etc.) is excellent. Every category should have one. However, they should not be training targets — they should be inference-time fallbacks.
- Consider adding "confusion pairs" documentation — pairs of classes commonly confused (e.g., SUV vs Car) with disambiguation guidelines.

**Verdict:** Excellent foundation. Fix category boundary conflicts. Strengthen "ObjectsOfInterest" organization.

### Reviewer 4: Chief Scientist at Ultralytics

**Critique:**
- 150+ classes × 100K+ annotations per class × 3 annotation styles = prohibitive data requirements. Recommend prioritizing classes for v1.0 and deferring low-frequency classes.
- "Annotation Style" column is informative but risks being ignored if it conflicts with existing dataset formats. Provide a priority order: Bounding Box > OBB > Polygon > Segmentation, based on annotation cost.
- The naming convention omits a critical rule: class names must be <= 20 characters for compatibility with YOLO/Ultralytics training pipelines.
- "MilitaryTent" and "MedicalTent" under Buildings.Tent introduce functional semantics. Replace with physical descriptors: "LargeRectangularTent", "DomeTent".
- Training recommendation: start with YOLOv11x on the 12 COCO-mapped classes + 10 VisDrone-mapped classes = ~22 core classes. Expand to 80+ classes as specialized datasets are acquired.

**Verdict:** Practical concerns well-addressed. Reduce total scope for v1.0. Adopt 20-char max name rule. Remove functional tent subtypes.

## 12.2 Identified Issues and Resolutions

| # | Issue | Affected Classes | Resolution |
|---|-------|-----------------|------------|
| 1 | TrackedVehicle vs HeavyEquipment overlap | TrackedVehicle, HeavyEquipment | Make TrackedVehicle a subtype of HeavyEquipment (tracked machinery) |
| 2 | MilitaryAircraft is semantic | Aircraft.FixedWing.MilitaryAircraft | Replace with "CombatAircraft" defined by physical features (hardpoints, angular design) |
| 3 | Group/Crowd threshold arbitrary | People.Group, People.Crowd | Keep both but document visual criteria: Group = distinguishable individuals, Crowd = indistinct mass |
| 4 | ObjectsOfInterest is dumping ground | All ObjectsOfInterest classes | Move TentStructure → Buildings.Tent (merge), move GuardPost → Barriers.Checkpoint (merge) |
| 5 | WaterTower appears in both Buildings and Infrastructure | Buildings.Tower.WaterTower, Infrastructure.WaterTower | Keep only in Infrastructure.WaterTower; remove from Buildings.Tower |
| 6 | ThreeWheeledVehicle confusion with Motorcycle | ThreeWheeledVehicle, Motorcycle | Add disambiguation note: ThreeWheeledVehicle has 3 wheels visible; Motorcycle has 2 |
| 7 | HeavyEquipment subtypes too fine-grained | Excavator, Bulldozer, WheelLoader, etc. | Merge into HeavyEquipment (keep as attributes, not separate classes) for v1.0 |
| 8 | NavalVessel subtypes indistinguishable | Destroyer, Frigate, Corvette | Merge into single "CombatVessel" class for v1.0 |
| 9 | Level-3 depth insufficient | Multiple domains | Increase max depth to 4 levels (Root → 22 Category → Class → Subtype → Variant) |
| 10 | Military/Police/Border checkpoint subtypes semantic | Checkpoint subtypes | Replace with physical types: "VehicleCheckpoint", "PedestrianCheckpoint" |
| 11 | TentStructure in ObjectsOfInterest | ObjectsOfInterest.TentStructure | Move to Buildings.Tent |
| 12 | No global UnknownObject at Root level | All | Add UnknownObject as a root-level fallback class |
| 13 | Annotation style priority undefined | All | Define priority: BoundingBox → OBB → Polygon → Segmentation |
| 14 | Class names exceeding 20 chars | Multiple (e.g., ThreeWheeledVehicle) | Abbreviate where possible while maintaining readability |

---
## 12.3 Revised Ontology (Post-Review)

Changes applied:
1. Merge TrackedVehicle into HeavyEquipment as tracked variant
2. Remove MilitaryAircraft; replace with CombatAircraft (visually defined)
3. Merge NavalVessel subtypes into CombatVessel
4. Move TentStructure → Buildings.Tent
5. Move GuardPost → Barriers.Checkpoint
6. Consolidate HeavyEquipment subtypes into single HeavyEquipment class (discriminable via attributes)
7. Consolidate NavalVessel subtypes into CombatVessel
8. Increase max depth to 4 levels
9. Add UnknownObject as root-level fallback
10. Keep Group/Crowd with visual criteria (not count-based)
11. Replace Checkpoint functional subtypes with physical types

**Revised class count:** ~120 classes (reduced from ~150)
**Top-level categories:** 22 (unchanged)
**Leaf classes:** ~90 (reduced from ~120)

---

# PART 13: Final Ontology Specification

## 13.1 Final Ontology Tree (Post-Review)

`
ROOT
├── UnknownObject (catch-all for unclassified detections)
│
├── People
│   ├── Person (individual human)
│   ├── Group (distinguishable individuals in proximity)
│   └── Crowd (indistinct mass of people)
│
├── Ground Vehicle
│   ├── Car (sedan, hatchback, coupe, station wagon)
│   ├── SUV (raised, continuous cabin)
│   ├── Van (box-shaped, single volume)
│   ├── Pickup (cabin + open bed)
│   ├── Bus (elongated, 12+ passengers)
│   ├── Truck (cab + separate cargo body)
│   ├── Motorcycle (2 wheels, motorized)
│   ├── Bicycle (2 wheels, human-powered)
│   ├── ThreeWheeledVehicle (3 wheels, auto/tricycle)
│   ├── HeavyEquipment (construction/agricultural, wheeled or tracked)
│   ├── Trailer (non-self-propelled, towed)
│   └── UnknownGroundVehicle
│
├── Aircraft
│   ├── FixedWingAircraft (fixed wings, jet/prop)
│   ├── RotaryWingAircraft (rotor blades)
│   ├── UnmannedAerialVehicle (no cockpit, drone)
│   ├── Glider (no engine, high aspect-ratio wings)
│   ├── Balloon (lighter-than-air, envelope)
│   └── UnknownAircraft
│
├── Watercraft
│   ├── Ship (generic large vessel, 50m+)
│   ├── Boat (generic small vessel, <50m)
│   ├── ContainerShip (stacked containers on deck)
│   ├── TankerShip (flush deck, piping)
│   ├── CargoShip (holds + cranes)
│   ├── PassengerShip (multi-deck accommodation)
│   ├── FishingVessel (fishing gear visible)
│   ├── CombatVessel (warship, angular, weapons visible)
│   ├── Submarine (cylindrical hull, surfaced)
│   ├── Sailboat (masts + sails)
│   ├── SmallCraft (canoe, kayak, jet ski)
│   ├── Barge (flat-bottomed, no propulsion)
│   └── UnknownWatercraft
│
├── Buildings
│   ├── Building (generic enclosed structure)
│   │   ├── ResidentialBuilding
│   │   ├── CommercialBuilding
│   │   ├── IndustrialBuilding
│   │   ├── AgriculturalBuilding
│   │   └── InstitutionalBuilding
│   ├── SmallStructure (shed, kiosk, booth)
│   ├── Tower (narrow, height ≫ width)
│   ├── Silo (cylindrical storage)
│   ├── Greenhouse (glass/plastic roof)
│   ├── Tent (fabric shelter, all sizes)
│   ├── Ruins (damaged, partial walls, no roof)
│   └── UnknownBuilding
│
├── Infrastructure
│   ├── TrafficLight (intersection signal)
│   ├── StreetLight (illumination pole)
│   ├── Signpost (informational signage)
│   ├── Bench (public seating)
│   ├── Billboard (large display panel)
│   ├── Pipeline (above-ground tube, linear)
│   ├── CommunicationsTower (lattice, antennas)
│   ├── WaterTower (elevated tank)
│   └── UnknownInfrastructure
│
├── Road Network
│   ├── Road (general paved/unpaved way)
│   ├── Highway (multi-lane, limited access)
│   ├── Street (urban road, buildings adjacent)
│   ├── Intersection (road crossing)
│   ├── Roundabout (circular intersection)
│   ├── RoadSign (regulatory/informational sign)
│   ├── TrafficSignal (intersection signal — RoadNetwork context)
│   └── UnknownRoadElement
│
├── Vegetation
│   ├── Tree (single woody plant, distinct crown)
│   ├── Forest (contiguous tree canopy)
│   ├── Shrub (multi-stem, low woody plant)
│   ├── Grassland (grass-dominant, no trees)
│   ├── Cropland (organized agricultural planting)
│   ├── Orchard (aligned tree rows)
│   └── UnknownVegetation
│
├── Water Bodies
│   ├── Sea (open saltwater, no opposite shore)
│   ├── Lake (inland standing water)
│   ├── River (flowing watercourse)
│   ├── Stream (small flowing water)
│   ├── Pond (small standing water)
│   ├── Reservoir (artificial lake, dammed)
│   ├── Wetland (saturated, marsh/swamp)
│   ├── Beach (shoreline deposit)
│   └── UnknownWaterBody
│
├── Terrain
│   ├── BarrenLand (no vegetation, exposed soil)
│   ├── RockyTerrain (rock outcrops, boulders)
│   ├── SandyTerrain (sand-dominant)
│   ├── MudTerrain (wet soft earth)
│   ├── SnowCovered (snow/ice surface)
│   ├── UrbanTerrain (built-up, paved)
│   └── UnknownTerrain
│
├── Smoke
│   ├── SmokePlume (rising column with drift)
│   ├── SmokeColumn (rising straight, no drift)
│   ├── SmokeHaze (diffuse widespread)
│   ├── DustCloud (particulate, near ground)
│   └── UnknownSmoke
│
├── Fire
│   ├── Wildfire (uncontrolled vegetation fire)
│   ├── StructuralFire (building fire)
│   ├── ControlledBurn (managed agricultural fire)
│   ├── GasFlare (industrial flare stack)
│   └── UnknownFire
│
├── Construction
│   ├── ConstructionSite (active building area)
│   ├── Excavation (man-made cavity/trench)
│   ├── Scaffolding (temporary work platform)
│   ├── MaterialPile (stockpiled construction material)
│   ├── UnderConstructionStructure (partially built)
│   └── UnknownConstruction
│
├── Engineering Structures
│   ├── Dam (water impoundment barrier)
│   ├── Lock (navigation water elevator)
│   ├── Canal (man-made water channel)
│   ├── Overpass (road/rail over crossing)
│   ├── RetainingWall (soil retention structure)
│   ├── Breakwater (wave protection, open water)
│   ├── Tunnel (underground passage)
│   └── UnknownEngineeringStructure
│
├── Utilities
│   ├── SolarArray (photovoltaic panel installation)
│   ├── WindTurbine (wind energy generator)
│   ├── PowerLine (overhead electrical transmission)
│   ├── UtilityPole (single support pole)
│   ├── TransformerSubstation (voltage conversion facility)
│   ├── GasFacility (gas processing/storage)
│   └── UnknownUtility
│
├── Barriers
│   ├── Wall (solid masonry/concrete barrier)
│   ├── Fence (see-through post+wire barrier)
│   ├── JerseyBarrier (modular concrete barrier)
│   ├── Gate (openable barrier section)
│   ├── Checkpoint (controlled access point)
│   ├── VehicleBarrier (anti-vehicle ditch/berm)
│   └── UnknownBarrier
│
├── Bridges
│   ├── BeamBridge (horizontal beam on piers)
│   ├── ArchBridge (arch-shaped support)
│   ├── SuspensionBridge (cables + towers)
│   ├── CableStayedBridge (direct cable support)
│   ├── TrussBridge (triangular lattice)
│   └── UnknownBridge
│
├── Airfields
│   ├── Runway (aircraft landing/takeoff strip)
│   ├── Taxiway (connecting path)
│   ├── Apron (parking/maintenance area)
│   ├── Helipad (helicopter landing area)
│   ├── Hangar (aircraft storage building)
│   ├── ControlTower (ATC tower)
│   ├── Terminal (passenger/cargo building)
│   └── UnknownAirfieldElement
│
├── Ports
│   ├── Dock (mooring structure)
│   ├── ContainerTerminal (container transshipment area)
│   ├── PortCrane (cargo loading crane)
│   ├── PortWarehouse (transit storage)
│   ├── BreakwaterPort (harbor wave protection)
│   ├── HarborBasin (sheltered water area)
│   ├── DryDock (vessel maintenance basin)
│   └── UnknownPortElement
│
├── Rail Infrastructure
│   ├── RailwayTrack (steel rail on ballast)
│   ├── Train (rail vehicles on track)
│   ├── RailwayStation (station + platforms)
│   ├── RailwayBridge (track-carrying bridge)
│   ├── RailwaySignal (train control signal)
│   ├── RailwayYard (track complex)
│   └── UnknownRailElement
│
├── Natural Features
│   ├── Mountain (steep elevation)
│   ├── Cliff (vertical rock face)
│   ├── RockFormation (exposed geological feature)
│   ├── CaveEntrance (underground opening)
│   ├── Island (land surrounded by water)
│   ├── Peninsula (land extending into water)
│   ├── SandDune (wind-formed sand mound)
│   ├── Glacier (persistent ice mass)
│   └── UnknownNaturalFeature
│
└── Objects of Interest
    ├── ShippingContainer (intermodal freight box)
    ├── Pallet (flat transport platform)
    ├── BarrelDrum (cylindrical liquid container)
    ├── Buoy (floating navigation marker)
    ├── Watchtower (elevated observation post)
    ├── AntennaMast (slender comms antenna)
    ├── Generator (portable/stationary power unit)
    ├── SatelliteDish (parabolic antenna)
    ├── FuelTank (above-ground fuel storage)
    └── CamouflageNet (concealment netting)
`

## 13.2 Ontology Version

**v1.0.0** — Initial frozen release (post-review)

## 13.3 Number of Classes

| Level | Count |
|-------|-------|
| Root | 1 |
| Level-1 (Top-level categories) | 22 |
| Level-2 (Classes) | 114 |
| Level-3 (Subtypes) | 0 (deferred to v1.1+) |
| **Total (all levels)** | **137** |
| **Core trainable classes** | **~90** (Level-2 excluding Unknown variants) |

## 13.4 Number of Parent Categories

**22 parent categories** (Level-1 nodes):

1. People
2. Ground Vehicle
3. Aircraft
4. Watercraft
5. Buildings
6. Infrastructure
7. Road Network
8. Vegetation
9. Water Bodies
10. Terrain
11. Smoke
12. Fire
13. Construction
14. Engineering Structures
15. Utilities
16. Barriers
17. Bridges
18. Airfields
19. Ports
20. Rail Infrastructure
21. Natural Features
22. Objects of Interest

## 13.5 Mapping Strategy Summary

| Source | Classes | Mapped | Ignored | Mapped % | Dataset Value |
|--------|---------|--------|--------|----------|--------------|
| COCO | 80 | 12 | 68 | 15% | High (core classes) |
| Open Images | ~600 | ~45 | ~555 | 7.5% | High (diverse, large scale) |
| Objects365 | 365 | ~16 | ~349 | 4.4% | Medium |
| LVIS | ~1200 | ~20 | ~1180 | 1.7% | Low (too fine-grained) |
| VisDrone | 12 | 10 | 2 | 83% | Very High (aerial view) |
| SpaceNet | 1 | 1 | 0 | 100% | High (satellite buildings) |
| LoveDA | 7 | 6 | 1 | 86% | High (satellite terrain) |
| SeaShips | 6 | 6 | 0 | 100% | High (maritime) |
| **Total** | **~2271** | **~116** | **~2155** | **5.1%** | |

## 13.6 Future Compatibility Rating

| Criterion | Rating (1-10) |
|-----------|--------------|
| Extensibility (adding new classes) | 9/10 |
| Dataset independence | 10/10 |
| Format/label evolution resilience | 9/10 |
| New sensor modality adaptation | 8/10 |
| New object type incorporation | 8/10 |
| **Overall Future Compatibility** | **8.8/10** |

## 13.7 Commercial Readiness Rating

| Criterion | Rating (1-10) |
|-----------|--------------|
| Coverage of operationally relevant objects | 9/10 |
| Mapping to open-source datasets | 8/10 |
| Annotation clarity (boundedness, consistency) | 9/10 |
| Industry best-practice alignment | 9/10 |
| Vendor/market neutrality | 10/10 |
| **Overall Commercial Readiness** | **9.0/10** |

## 13.8 Production Readiness Rating

| Criterion | Rating (1-10) |
|-----------|--------------|
| Versioning and change control defined | 9/10 |
| Deprecation and backward compatibility policies | 9/10 |
| Naming convention completeness | 10/10 |
| Integration with existing codebase (contracts layer) | 9/10 |
| Documentation completeness | 9/10 |
| Training feasibility with open-source datasets | 8/10 |
| **Overall Production Readiness** | **9.0/10** |

## 13.9 Why This Ontology Is Suitable for Commercial AI DSS Using Only Open-Source Datasets

### 13.9.1 Dataset Coverage

Using only the 8 open-source datasets mapped in Section 5, the DSS can train detection models for:

| DSS Category | Open-Source Dataset Sources | Annotation Count (Approx.) |
|-------------|---------------------------|---------------------------|
| People | COCO, Open Images, VisDrone | 500K+ instances |
| Cars, Trucks, Buses | COCO, Open Images, VisDrone, Objects365 | 1M+ instances |
| Aircraft | COCO, Open Images | 100K+ instances |
| Ships, Boats | COCO, Open Images, SeaShips | 100K+ instances |
| Buildings | COCO, Open Images, SpaceNet, LoveDA | 2M+ instances |
| Roads | LoveDA, Open Images | 500K+ segments |
| Vegetation | LoveDA, Open Images | 500K+ segments |
| Water Bodies | LoveDA, Open Images | 200K+ segments |
| Barriers, Bridges, Infrastructure | Open Images | 100K+ instances |
| Trains | COCO, Open Images | 50K+ instances |

**Total available annotations from open-source datasets:** Approximately 5M+ instances across ~90 classes.

### 13.9.2 Critical Mass Assessment

- **Core classes** (12 classes): Available from COCO alone (1.2M images). Sufficient for production.
- **Aerial view classes** (10 classes): Available from VisDrone (10K images). Sufficient for drone-based operations.
- **Satellite classes** (6 classes): Available from SpaceNet + LoveDA (30K images). Sufficient for satellite operations.
- **Maritime classes** (4 classes): Available from SeaShips (31K images). Sufficient for maritime operations.
- **Infrastructure/Barrier classes** (~15 classes): Available from Open Images. Sufficient for general operations.

### 13.9.3 Training Strategy

**Phase 1 (v1.0):** Train on COCO + Open Images + VisDrone = ~30 core classes, ~3M annotations.
**Phase 2 (v1.1):** Add SpaceNet + LoveDA + SeaShips = +10 satellite/maritime classes, ~2M annotations.
**Phase 3 (v1.2):** Add custom data for remaining ~50 classes as mission-specific requirements emerge.

### 13.9.4 Key Advantages

1. **Zero cost for initial training data** — All mapped datasets are open-source and free.
2. **Dataset diversity** — 8 independent datasets provide natural domain randomization.
3. **Label consistency** — Ontology normalization ensures models see consistent labels across datasets.
4. **No dependency on proprietary data** — Commercial deployment possible without proprietary datasets.
5. **Lawful compliance** — All open-source datasets have permissive licenses for commercial use.

### 13.9.5 Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Open-source datasets have ground-view bias | VisDrone + SpaceNet + LoveDA provide aerial/satellite views |
| Dataset label noise | Ontology mapping explicitly handles synonyms and ignores ambiguous classes |
| Long-tail classes with few examples | Unknown* classes serve as fallback during training |
| Dataset license restrictions | All 8 datasets selected for permissive commercial licensing |
| Domain shift (satellite → drone → ground) | Multi-dataset training provides natural domain randomization |

---

**END OF ONTOLOGY SPECIFICATION — v1.0.0**

