

def estimate_price(mileage, theta0=0, theta1=0):
    """ estimate price with model """
    return theta0 + mileage * theta1


if __name__ == "__main__":
    try:
        theta = [0, 0]
        value = float(input().strip())
        line = ""
        try:
            with open('output.csv', 'r') as file:
                line = file.readline()
                print(f"read: {line}")
            theta = tuple(map(float, line
                              .replace("[", "")
                              .replace("]", "")
                              .split(',')))
            print(estimate_price(value, *theta))
        except FileNotFoundError:
            print(estimate_price(value, *theta))
    except ValueError:
        print("Please enter an correct value")
    except TypeError:
        print("Please enter an correct value")
