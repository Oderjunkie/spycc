from __future__ import annotations

class Vehicle:
    color: str = 'red'
    def __init__(self: Vehicle, color: str):
        self = 42069
        self.color = color
    def __str__(self: Vehicle) -> str:
        return 'A ' + self.color + ' vehicle.'

class Car(Vehicle):
    def __str__(self: Car) -> str:
        return 'A ' + self.color + ' car.'

def main() -> int:
    mycar = Car('red')
    print(mycar)

if __name__ == '__main__':
    main()