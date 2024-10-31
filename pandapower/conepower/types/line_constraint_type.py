from enum import Enum


class LineConstraintType(Enum):
    UNKNOWN = 0
    APPARENT_POWER = 1
    CURRENT = 2

    @classmethod
    def from_str(cls, string):
        if string == 'S':
            return cls.APPARENT_POWER
        elif string == 'I':
            return cls.CURRENT

        return cls.UNKNOWN
