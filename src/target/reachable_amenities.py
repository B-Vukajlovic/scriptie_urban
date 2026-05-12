"""Compute amenity reachability metrics by category and radius."""

from __future__ import annotations

from scipy.spatial import KDTree
import numpy as np
import pandas as pd


AMENITY_TO_CATEGORY: dict[str, str] = {
	"restaurant": "food_retail",
	"cafe": "food_retail",
	"fast_food": "food_retail",
	"bar": "food_retail",
	"pub": "food_retail",
	"marketplace": "food_retail",
	"supermarket": "food_retail",
	"convenience": "food_retail",
	"greengrocer": "food_retail",
	"bakery": "food_retail",
	"butcher": "food_retail",
	"deli": "food_retail",
	"hospital": "healthcare",
	"clinic": "healthcare",
	"doctors": "healthcare",
	"dentist": "healthcare",
	"pharmacy": "healthcare",
	"veterinary": "healthcare",
	"school": "education",
	"college": "education",
	"university": "education",
	"kindergarten": "education",
	"library": "education",
	"park": "recreation",
	"sports_centre": "recreation",
	"post_office": "public_services",
	"courthouse": "public_services",
	"police": "public_services",
	"fire_station": "public_services",
	"townhall": "public_services",
	"community_centre": "public_services",
}


SHOP_TO_CATEGORY: dict[str, str] = {
	"supermarket": "food_retail",
	"convenience": "food_retail",
	"greengrocer": "food_retail",
	"bakery": "food_retail",
	"butcher": "food_retail",
	"deli": "food_retail",
	"seafood": "food_retail",
	"cheese": "food_retail",
	"farm": "food_retail",
	"alcohol": "food_retail",
	"beverages": "food_retail",
	"department_store": "food_retail",
	"mall": "food_retail",
}


LEISURE_TO_CATEGORY: dict[str, str] = {
	"park": "recreation",
	"sports_centre": "recreation",
	"playground": "recreation",
	"pitch": "recreation",
	"stadium": "recreation",
	"fitness_centre": "recreation",
}


def map_amenity_category(amenity: pd.Series, shop: pd.Series, leisure: pd.Series) -> pd.Series:
	"""Map raw OSM tags to thesis amenity categories."""
	amenity_l = amenity.fillna("").str.lower()
	shop_l = shop.fillna("").str.lower()
	leisure_l = leisure.fillna("").str.lower()

	category = amenity_l.map(AMENITY_TO_CATEGORY).astype("object")
	category.loc[category.isna()] = shop_l[category.isna()].map(SHOP_TO_CATEGORY)
	category.loc[category.isna()] = leisure_l[category.isna()].map(LEISURE_TO_CATEGORY)
	return category


def amenity_counts_by_radius(
	origins_xy: np.ndarray,
	amenities_xy: np.ndarray,
	amenity_categories: pd.Series,
	radii_m: list[float],
	categories: list[str],
) -> pd.DataFrame:
	"""Count reachable amenities by category and radius for each origin."""
	out = pd.DataFrame(index=np.arange(len(origins_xy)))
	if len(amenities_xy) == 0:
		for category in categories:
			for r in radii_m:
				out[f"amen_{category}_{int(r/1000)}km"] = 0.0
		return out

	for category in categories:
		mask = amenity_categories.fillna("") == category
		cat_points = amenities_xy[mask.to_numpy()]
		if len(cat_points) == 0:
			for r in radii_m:
				out[f"amen_{category}_{int(r/1000)}km"] = 0.0
			continue

		tree = KDTree(cat_points)
		for r in radii_m:
			neighbors = tree.query_ball_point(origins_xy, r=float(r))
			counts = np.array([len(idx) for idx in neighbors], dtype=float)
			out[f"amen_{category}_{int(r/1000)}km"] = counts

	return out.reset_index(drop=True)
