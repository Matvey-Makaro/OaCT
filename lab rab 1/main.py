import tests


def main():
    tests.stress_test(10)
    tests.test_congenital_matrix()
    tests.test_simple_matrix()


if __name__ == '__main__':
    main()
