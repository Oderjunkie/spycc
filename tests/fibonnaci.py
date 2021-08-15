def fib(n: int) -> int:
    if n <= 1:
        return 1
    return fib(n - 1) + fib(n - 2)

def main() -> int:
    print('FIB 1', fib(1), 'FIB 2', fib(2), 'FIB 3', fib(3), 'FIB 4', fib(4), 'FIB 5', fib(5), sep='\n')

if __name__ == '__main__':
    main()