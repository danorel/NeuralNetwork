from typing import List, Tuple

RED = 2
GREEN = 1
BLUE = 0

raw_data: List[Tuple[List[int], int]] = [
    ([179, 127, 179], 2),
    ([102, 183, 163], 1),
    ([80, 152, 227], 0),
    ([184, 203, 172], 1),
    ([105, 165, 96], 1),
    ([30, 30, 30], 0),
    ([51, 85, 118], 0),
]

normalised_data: List[Tuple[List[float], int]] = [
    ([code / 255 for code in raw_record[0]], raw_record[1])
    for raw_record in raw_data
]

processed_data: List[Tuple[float, int]] = [
    (sum(normalised_record[0]) / 3, normalised_record[1]) for normalised_record in normalised_data
]
