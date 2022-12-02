from datetime import datetime


class People:
    def __init__(self, date: datetime, population_shift: int):
        self.date = date
        self.population_shift = population_shift
