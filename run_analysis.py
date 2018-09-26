# driver file to run the omega_a analysis
# configured using config files
#
# Aaron Fienberg
# September 2018


from precessionlib import analysis
import json


def main():
    with open('conf60H.json') as file:
        config = json.load(file)

    analysis.run_analysis(config)


if __name__ == '__main__':
    main()
