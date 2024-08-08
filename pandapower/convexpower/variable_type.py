from enum import Enum


class VariableType(Enum):
    UNKNOWN = 0
    UMAG = 1
    UANG = 2
    PG = 3
    QG = 4
    CJJ = 5
    CJK = 6
    SJK = 7

    @classmethod
    def from_str(cls, string):
        if string == 'Vm':
            return cls.UMAG
        elif string == 'Va':
            return cls.UANG
        elif string == 'Pg':
            return cls.PG
        elif string == 'Qg':
            return cls.QG

        return cls.UNKNOWN
