"""Core enumerations shared across every DSS module.

PERCEPTION LAYER — Computer Vision Ontology (v1.0.0)
=====================================================
These enums are the IMMUTABLE interface between the Computer
Vision module and all downstream modules. They describe only
physical objects — never intent, affiliation, or threat.

SEMANTIC LAYER — Knowledge / Fusion / Decision
===============================================
Nationality, ThreatLevel, TerrainType, and DecisionStatus are
owned by the Knowledge and Fusion engines, not by CV. They are
kept here for convenience but must NOT appear in CV output.
"""

from enum import Enum

# =============================================================================
# PERCEPTION LAYER — Immutable Computer Vision Ontology (v1.0.0)
# =============================================================================


class ObjectType(str, Enum):
    """Perception-only classification of objects detected in imagery.

    Computer Vision answers ONLY: "What physically exists in this image?"
    It NEVER answers: "What does it mean?" — that is the Knowledge Engine's job.

    Naming convention: PascalCase. Unused here (str enum requires flat values)
    but the dotted-path convention is preserved in the value string.
    """

    # --- Root fallback ---
    UNKNOWN_OBJECT = "unknown_object"

    # --- People ---
    PEOPLE_PERSON = "people.person"
    PEOPLE_GROUP = "people.group"
    PEOPLE_CROWD = "people.crowd"

    # --- Ground Vehicle ---
    GROUND_VEHICLE_CAR = "ground_vehicle.car"
    GROUND_VEHICLE_SUV = "ground_vehicle.suv"
    GROUND_VEHICLE_VAN = "ground_vehicle.van"
    GROUND_VEHICLE_PICKUP = "ground_vehicle.pickup"
    GROUND_VEHICLE_BUS = "ground_vehicle.bus"
    GROUND_VEHICLE_TRUCK = "ground_vehicle.truck"
    GROUND_VEHICLE_MOTORCYCLE = "ground_vehicle.motorcycle"
    GROUND_VEHICLE_BICYCLE = "ground_vehicle.bicycle"
    GROUND_VEHICLE_THREE_WHEELED = "ground_vehicle.three_wheeled"
    GROUND_VEHICLE_HEAVY_EQUIPMENT = "ground_vehicle.heavy_equipment"
    GROUND_VEHICLE_TRAILER = "ground_vehicle.trailer"
    GROUND_VEHICLE_UNKNOWN = "ground_vehicle.unknown"

    # --- Aircraft ---
    AIRCRAFT_FIXED_WING = "aircraft.fixed_wing"
    AIRCRAFT_ROTARY_WING = "aircraft.rotary_wing"
    AIRCRAFT_UAV = "aircraft.uav"
    AIRCRAFT_GLIDER = "aircraft.glider"
    AIRCRAFT_BALLOON = "aircraft.balloon"
    AIRCRAFT_UNKNOWN = "aircraft.unknown"

    # --- Watercraft ---
    WATERCRAFT_SHIP = "watercraft.ship"
    WATERCRAFT_BOAT = "watercraft.boat"
    WATERCRAFT_CONTAINER_SHIP = "watercraft.container_ship"
    WATERCRAFT_TANKER_SHIP = "watercraft.tanker_ship"
    WATERCRAFT_CARGO_SHIP = "watercraft.cargo_ship"
    WATERCRAFT_PASSENGER_SHIP = "watercraft.passenger_ship"
    WATERCRAFT_FISHING_VESSEL = "watercraft.fishing_vessel"
    WATERCRAFT_COMBAT_VESSEL = "watercraft.combat_vessel"
    WATERCRAFT_SUBMARINE = "watercraft.submarine"
    WATERCRAFT_SAILBOAT = "watercraft.sailboat"
    WATERCRAFT_SMALL_CRAFT = "watercraft.small_craft"
    WATERCRAFT_BARGE = "watercraft.barge"
    WATERCRAFT_UNKNOWN = "watercraft.unknown"

    # --- Buildings ---
    BUILDINGS_BUILDING = "buildings.building"
    BUILDINGS_SMALL_STRUCTURE = "buildings.small_structure"
    BUILDINGS_TOWER = "buildings.tower"
    BUILDINGS_SILO = "buildings.silo"
    BUILDINGS_GREENHOUSE = "buildings.greenhouse"
    BUILDINGS_TENT = "buildings.tent"
    BUILDINGS_RUINS = "buildings.ruins"
    BUILDINGS_UNKNOWN = "buildings.unknown"

    # --- Infrastructure ---
    INFRASTRUCTURE_TRAFFIC_LIGHT = "infrastructure.traffic_light"
    INFRASTRUCTURE_STREET_LIGHT = "infrastructure.street_light"
    INFRASTRUCTURE_SIGNPOST = "infrastructure.signpost"
    INFRASTRUCTURE_BENCH = "infrastructure.bench"
    INFRASTRUCTURE_BILLBOARD = "infrastructure.billboard"
    INFRASTRUCTURE_PIPELINE = "infrastructure.pipeline"
    INFRASTRUCTURE_COMMS_TOWER = "infrastructure.comms_tower"
    INFRASTRUCTURE_WATER_TOWER = "infrastructure.water_tower"
    INFRASTRUCTURE_UNKNOWN = "infrastructure.unknown"

    # --- Road Network ---
    ROAD_NETWORK_ROAD = "road_network.road"
    ROAD_NETWORK_HIGHWAY = "road_network.highway"
    ROAD_NETWORK_STREET = "road_network.street"
    ROAD_NETWORK_INTERSECTION = "road_network.intersection"
    ROAD_NETWORK_ROUNDABOUT = "road_network.roundabout"
    ROAD_NETWORK_ROAD_SIGN = "road_network.road_sign"
    ROAD_NETWORK_TRAFFIC_SIGNAL = "road_network.traffic_signal"
    ROAD_NETWORK_UNKNOWN = "road_network.unknown"

    # --- Vegetation ---
    VEGETATION_TREE = "vegetation.tree"
    VEGETATION_FOREST = "vegetation.forest"
    VEGETATION_SHRUB = "vegetation.shrub"
    VEGETATION_GRASSLAND = "vegetation.grassland"
    VEGETATION_CROPLAND = "vegetation.cropland"
    VEGETATION_ORCHARD = "vegetation.orchard"
    VEGETATION_UNKNOWN = "vegetation.unknown"

    # --- Water Bodies ---
    WATER_BODIES_SEA = "water_bodies.sea"
    WATER_BODIES_LAKE = "water_bodies.lake"
    WATER_BODIES_RIVER = "water_bodies.river"
    WATER_BODIES_STREAM = "water_bodies.stream"
    WATER_BODIES_POND = "water_bodies.pond"
    WATER_BODIES_RESERVOIR = "water_bodies.reservoir"
    WATER_BODIES_WETLAND = "water_bodies.wetland"
    WATER_BODIES_BEACH = "water_bodies.beach"
    WATER_BODIES_UNKNOWN = "water_bodies.unknown"

    # --- Terrain ---
    TERRAIN_BARREN = "terrain.barren"
    TERRAIN_ROCKY = "terrain.rocky"
    TERRAIN_SANDY = "terrain.sandy"
    TERRAIN_MUD = "terrain.mud"
    TERRAIN_SNOW_COVERED = "terrain.snow_covered"
    TERRAIN_URBAN = "terrain.urban"
    TERRAIN_UNKNOWN = "terrain.unknown"

    # --- Smoke ---
    SMOKE_PLUME = "smoke.plume"
    SMOKE_COLUMN = "smoke.column"
    SMOKE_HAZE = "smoke.haze"
    SMOKE_DUST_CLOUD = "smoke.dust_cloud"
    SMOKE_UNKNOWN = "smoke.unknown"

    # --- Fire ---
    FIRE_WILDFIRE = "fire.wildfire"
    FIRE_STRUCTURAL = "fire.structural"
    FIRE_CONTROLLED_BURN = "fire.controlled_burn"
    FIRE_GAS_FLARE = "fire.gas_flare"
    FIRE_UNKNOWN = "fire.unknown"

    # --- Construction ---
    CONSTRUCTION_SITE = "construction.site"
    CONSTRUCTION_EXCAVATION = "construction.excavation"
    CONSTRUCTION_SCAFFOLDING = "construction.scaffolding"
    CONSTRUCTION_MATERIAL_PILE = "construction.material_pile"
    CONSTRUCTION_UNDER_CONSTRUCTION = "construction.under_construction"
    CONSTRUCTION_UNKNOWN = "construction.unknown"

    # --- Engineering Structures ---
    ENGINEERING_DAM = "engineering.dam"
    ENGINEERING_LOCK = "engineering.lock"
    ENGINEERING_CANAL = "engineering.canal"
    ENGINEERING_OVER_PASS = "engineering.overpass"
    ENGINEERING_RETAINING_WALL = "engineering.retaining_wall"
    ENGINEERING_BREAKWATER = "engineering.breakwater"
    ENGINEERING_TUNNEL = "engineering.tunnel"
    ENGINEERING_UNKNOWN = "engineering.unknown"

    # --- Utilities ---
    UTILITIES_SOLAR_ARRAY = "utilities.solar_array"
    UTILITIES_WIND_TURBINE = "utilities.wind_turbine"
    UTILITIES_POWER_LINE = "utilities.power_line"
    UTILITIES_UTILITY_POLE = "utilities.utility_pole"
    UTILITIES_TRANSFORMER_SUBSTATION = "utilities.transformer_substation"
    UTILITIES_GAS_FACILITY = "utilities.gas_facility"
    UTILITIES_UNKNOWN = "utilities.unknown"

    # --- Barriers ---
    BARRIERS_WALL = "barriers.wall"
    BARRIERS_FENCE = "barriers.fence"
    BARRIERS_JERSEY = "barriers.jersey"
    BARRIERS_GATE = "barriers.gate"
    BARRIERS_CHECKPOINT = "barriers.checkpoint"
    BARRIERS_VEHICLE_BARRIER = "barriers.vehicle_barrier"
    BARRIERS_UNKNOWN = "barriers.unknown"

    # --- Bridges ---
    BRIDGES_BEAM = "bridges.beam"
    BRIDGES_ARCH = "bridges.arch"
    BRIDGES_SUSPENSION = "bridges.suspension"
    BRIDGES_CABLE_STAYED = "bridges.cable_stayed"
    BRIDGES_TRUSS = "bridges.truss"
    BRIDGES_UNKNOWN = "bridges.unknown"

    # --- Airfields ---
    AIRFIELDS_RUNWAY = "airfields.runway"
    AIRFIELDS_TAXIWAY = "airfields.taxiway"
    AIRFIELDS_APRON = "airfields.apron"
    AIRFIELDS_HELIPAD = "airfields.helipad"
    AIRFIELDS_HANGAR = "airfields.hangar"
    AIRFIELDS_CONTROL_TOWER = "airfields.control_tower"
    AIRFIELDS_TERMINAL = "airfields.terminal"
    AIRFIELDS_UNKNOWN = "airfields.unknown"

    # --- Ports ---
    PORTS_DOCK = "ports.dock"
    PORTS_CONTAINER_TERMINAL = "ports.container_terminal"
    PORTS_CRANE = "ports.crane"
    PORTS_WAREHOUSE = "ports.warehouse"
    PORTS_BREAKWATER = "ports.breakwater"
    PORTS_HARBOR_BASIN = "ports.harbor_basin"
    PORTS_DRY_DOCK = "ports.dry_dock"
    PORTS_UNKNOWN = "ports.unknown"

    # --- Rail Infrastructure ---
    RAIL_TRACK = "rail.track"
    RAIL_TRAIN = "rail.train"
    RAIL_STATION = "rail.station"
    RAIL_BRIDGE = "rail.bridge"
    RAIL_SIGNAL = "rail.signal"
    RAIL_YARD = "rail.yard"
    RAIL_UNKNOWN = "rail.unknown"

    # --- Natural Features ---
    NATURAL_MOUNTAIN = "natural.mountain"
    NATURAL_CLIFF = "natural.cliff"
    NATURAL_ROCK_FORMATION = "natural.rock_formation"
    NATURAL_CAVE_ENTRANCE = "natural.cave_entrance"
    NATURAL_ISLAND = "natural.island"
    NATURAL_PENINSULA = "natural.peninsula"
    NATURAL_SAND_DUNE = "natural.sand_dune"
    NATURAL_GLACIER = "natural.glacier"
    NATURAL_UNKNOWN = "natural.unknown"

    # --- Military ---
    MILITARY_ARTILLERY = "artillery"
    MILITARY_MISSILE = "missile"
    MILITARY_TANK = "tank"
    MILITARY_ARMORED_VEHICLE = "armored_vehicle"

    # --- Objects of Interest ---
    OOI_SHIPPING_CONTAINER = "ooi.shipping_container"
    OOI_PALLET = "ooi.pallet"
    OOI_BARREL_DRUM = "ooi.barrel_drum"
    OOI_BUOY = "ooi.buoy"
    OOI_WATCHTOWER = "ooi.watchtower"
    OOI_ANTENNA_MAST = "ooi.antenna_mast"
    OOI_GENERATOR = "ooi.generator"
    OOI_SATELLITE_DISH = "ooi.satellite_dish"
    OOI_FUEL_TANK = "ooi.fuel_tank"
    OOI_CAMOUFLAGE_NET = "ooi.camouflage_net"


# =============================================================================
# SEMANTIC LAYER — Belongs to Knowledge / Fusion / Decision (not CV output)
# =============================================================================


class ThreatLevel(str, Enum):
    """Severity of an identified threat.

    Owned by the Fusion Engine. CV must NEVER output this.
    """

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Nationality(str, Enum):
    """Affiliation of a detected entity.

    Owned by the Knowledge Engine (Friendly / Enemy modules).
    CV must NEVER output this.
    """

    UNKNOWN = "unknown"
    FRIENDLY = "friendly"
    HOSTILE = "hostile"
    CIVILIAN = "civilian"
    NEUTRAL = "neutral"


class TerrainType(str, Enum):
    """Classification of terrain features.

    Owned by the Terrain Knowledge module.
    CV must NEVER output this — CV detects physical objects only.
    """

    ROAD = "road"
    FOREST = "forest"
    RIVER = "river"
    HILL = "hill"
    OPEN_FIELD = "open_field"
    URBAN = "urban"
    BRIDGE = "bridge"
    UNKNOWN = "unknown"


class DecisionStatus(str, Enum):
    """Lifecycle state of a decision recommendation.

    Owned by the Decision Engine.
    """

    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    RECOMMENDED = "recommended"
    APPROVED = "approved"
    REJECTED = "rejected"
