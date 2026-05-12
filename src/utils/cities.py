"""Supported city metadata shared by pipeline scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CityConfig:
    state: str
    place_geoid: str


CITY_CONFIGS: dict[str, CityConfig] = {
    "santa_cruz": CityConfig(state="ca", place_geoid="0669112"),
    "little_rock": CityConfig(state="ar", place_geoid="0541000"),
    "tulsa": CityConfig(state="ok", place_geoid="4075000"),
    "denver": CityConfig(state="co", place_geoid="0820000"),
    "atlanta": CityConfig(state="ga", place_geoid="1304000"),
    "boston": CityConfig(state="ma", place_geoid="2507000"),
    "washington_dc": CityConfig(state="dc", place_geoid="1150000"),
    "houston": CityConfig(state="tx", place_geoid="4835000"),
    "los_angeles": CityConfig(state="ca", place_geoid="0644000"),
    "new_york": CityConfig(state="ny", place_geoid="3651000"),
}

DEFAULT_CITIES = list(CITY_CONFIGS)


def normalize_city(city: str) -> str:
    return city.strip().lower()


def validate_cities(cities: list[str]) -> list[str]:
    normalized = [normalize_city(city) for city in cities]
    unknown = [city for city in normalized if city not in CITY_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown cities: {unknown}. Supported: {DEFAULT_CITIES}")
    return normalized


def resolve_city_inputs(
    city: str,
    state: str | None = None,
    place_geoid: str | None = None,
) -> tuple[str, str]:
    city = normalize_city(city)
    defaults = CITY_CONFIGS.get(city)
    state_code = state or (defaults.state if defaults else None)
    place_id = place_geoid or (defaults.place_geoid if defaults else None)
    if not state_code or not place_id:
        raise ValueError(
            "Missing city metadata. Provide both --state and --place-geoid, "
            f"or use one of the supported cities: {DEFAULT_CITIES}."
        )
    return state_code.lower(), place_id


def resolve_state(city: str, state: str | None = None) -> str:
    city = normalize_city(city)
    if state:
        return state.lower()
    config = CITY_CONFIGS.get(city)
    if config is None:
        raise ValueError(f"Unknown city '{city}', provide --state explicitly.")
    return config.state
