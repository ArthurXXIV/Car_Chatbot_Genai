import re
import json
import logging
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import TimeoutException, NoSuchElementException


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


brands = [
    'maruti-suzuki-cars',
    'tata-cars',
    'kia-cars',
    'toyota-cars',
    'bmw-cars',
    'hyundai-cars',
    'mahindra-cars',
    'honda-cars',
    'mg-cars',
    'skoda-cars',
    'jaguar-cars',
    'audi-cars',
    'jeep-cars',
    'renault-cars',
    'porsche-cars',
    'nissan-cars',
    'rolls-royce-cars',
    'byd-cars',
    'citroen-cars',
    'lamborghini-cars',
    'volvo-cars',
    'ferrari-cars',
    'ford-cars',
    'lexus-cars',
    'bugatti-cars',
    'tesla-cars',
    'volkswagen-cars',
    'bentley-cars',
    'isuzu-cars',
    'lotus-cars',
    'maserati-cars',
    'mini-cars',
    'aston-martin-cars',
    'mclaren-cars',
    'mitsubishi-cars',
    'land-rover-cars',
    'haval-cars',
    'ora-cars',
    'peugeot-cars',
    'mercedes-benz-cars',
    'fisker-cars',
    'force-motors-cars',
    'pmv-cars',
    'pravaig-cars']


def feature_extract(df, feature):
    feature_values = df['Features'].str.extract(rf'{feature}\n(.*)\n')
    return feature_values


if __name__ == '__main__':
    options = webdriver.ChromeOptions()
    # options.add_argument('--start-maximized')
    options.add_argument("--log-level=3")
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)

    brand_bar = tqdm(total=len(brands), position=0, leave=True, desc="Brands")
    cars_bar = None
    variants_bar = None
    data = {}
    for brand in brands:
        try:
            driver.get('https://www.carwale.com/'+brand)
            brand_bar.set_description(f'Processing Brand: {brand}')
            data[brand] = {}
            cars = [a for a in WebDriverWait(driver, 10).until(ec.presence_of_all_elements_located(
                (By.CSS_SELECTOR, 'a[class="o-cpnuEd o-SoIQT o-cJrNdO o-fzpilz"]')))]
            cars_names = [car.text for car in cars]
            cars_links = [car.get_attribute('href') for car in cars]
            if cars_bar is None:
                cars_bar = tqdm(total=len(cars_names), position=1,
                                leave=False, desc="Cars")
            else:
                cars_bar.total = len(cars_names)
                cars_bar.reset()
            for car in zip(cars_names, cars_links):
                try:
                    cars_bar.set_description(f'Processing Car: {car[0]}')
                    data[brand][car[0]] = {}
                    driver.get(car[1])
                    try:
                        WebDriverWait(driver, 5).until(ec.presence_of_element_located(
                            (By.CSS_SELECTOR, 'div[class="o-fzptVd o-fzptYr o-frwuxB o-tvvmc o-ccrPDo o-bkmzIL o-djSZRV o-eCFISO o-eFudgX"]'))).click()
                    except:
                        pass

                    variants = [a for a in WebDriverWait(driver, 10).until(ec.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, 'a[class="o-cJrNdO o-jjpuv o-cVMLxW "]')))]
                    variants_names = [variant.text for variant in variants]
                    variants_links = [variant.get_attribute(
                        'href') for variant in variants]
                    if variants_bar is None:
                        variants_bar = tqdm(
                            total=len(variants_names), position=2, leave=False, desc="Variants")
                    else:
                        variants_bar.total = len(variants_names)
                        variants_bar.reset()
                    for variant in zip(variants_names, variants_links):
                        try:
                            variants_bar.set_description(
                                f'Variant: {variant[0]}')
                            driver.get(variant[1])

                            # try:
                            #     WebDriverWait(driver, 5).until(ec.presence_of_element_located(
                            #         (By.CSS_SELECTOR, 'div[aria-label="Read More"]'))).click()
                            # except Exception as e:
                            #     pass
                            for i in range(3):
                                try:
                                    description = WebDriverWait(driver, 5).until(ec.presence_of_element_located(
                                        (By.CSS_SELECTOR, 'div[data-lang-id="model_summary_content"]'))).text
                                except Exception as e:
                                    if i == 2:
                                        description = None
                                        logging.error(f"Error in description for brand: {brand}, car: {car[0]} and variant: {variant[0]} | Message: \n{e}")
                                        continue

                            try:
                                price = WebDriverWait(driver, 2).until(ec.presence_of_element_located(
                                    (By.CSS_SELECTOR, 'span[class="o-Hyyko o-bPYcRG o-eqqVmt"]'))).text

                            except TimeoutException:

                                price = WebDriverWait(driver, 2).until(ec.presence_of_element_located(
                                    (By.CSS_SELECTOR, 'span[class="o-eqqVmt o-Hyyko o-bPYcRG"]'))).text
                            except Exception as e:
                                price = None
                                logging.error(f"Error in price for brand: {brand}, car: {car[0]} and variant: {variant[0]} | Message: \n{e}")
                                continue

                            try:
                                details = WebDriverWait(driver, 10).until(
                                    ec.presence_of_element_located((By.CSS_SELECTOR, 'div[data-index="0"]'))).get_attribute('innerText')
                            except Exception as e:
                                details = None
                                logging.error(f"Error in details for brand: {brand}, car: {car[0]} and variant: {variant[0]} | Message: \n{e}")
                                continue
                            try:
                                features = WebDriverWait(driver, 10).until(
                                    ec.presence_of_element_located((By.CSS_SELECTOR, 'div[data-index="1"]'))).get_attribute('innerText')
                            except Exception as e:
                                features = None
                                logging.error(f"Error in details for brand: {brand}, car: {car[0]} and variant: {variant[0]} | Message: \n{e}")
                                continue

                            data[brand][car[0]][variant[0]] = {
                                'description': description,
                                'price': price,
                                'details': details,
                                'features': features
                            }
                        except Exception as e:
                            data[brand][car[0]][variant[0]] = '?'
                            print(
                                f"Error for brand: {brand}, car: {car[0]} and variant: {variant[0]} | Message: \n{e}")
                        finally:
                            variants_bar.update(1)
                except Exception as e:
                    data[brand][car[0]] = '?'
                    logging.error(f"Error for brand: {brand} and car: {car[0]} | Message: \n{e}")
                finally:
                    cars_bar.update(1)
        except Exception as e:
            data[brand] = '?'
            logging.error(f"Error for {brand} | Message: \n{e}")
        finally:
            brand_bar.update(1)
    brand_bar.close()
    if cars_bar is not None:
        cars_bar.close()
    if variants_bar is not None:
        variants_bar.close()
    logging.info("All Data Collected")

    with open('car_data.json', 'w') as f:
        json.dump(data, f)

    cars = []
    variants = []
    brands = []
    descriptions = []
    prices = []
    details = []
    features = []

    for brand, cars_data in data.items():
        if type(cars_data) == dict:
            for car, variants_data in cars_data.items():
                if type(variants_data) == dict:
                    for variant, data in variants_data.items():
                        if type(data) == dict:
                            cars.append(car)
                            variants.append(variant)
                            brands.append(brand)
                            descriptions.append(data['description'])
                            prices.append(data['price'])
                            details.append(data['details'])
                            features.append(data['features'])

    df = pd.DataFrame({
        'Brand': brands,
        'Car': cars,
        'Variant': variants,
        'Description': descriptions,
        'Price': prices,
        'Details': details,
        'Features': features
    })

    # df['Fuel Type'] = df['Details'].str.extract(r'\nFuel Type\n(.*)\n')
    # df['Mileage'] = df['Details'].str.extract(r'\nMileage \(ARAI\)\n(.*)\n')
    # df['Transmission'] = df['Details'].str.extract(r'\nTransmission\n(.*)\n')
    # df['Engine'] = df['Details'].str.extract(r'\nEngine\n(.*)\n')

    # df.drop('Details', axis=1, inplace=True)

    # safety_features = df['Features'][0].split('\n\nSafety')[1].split(
    #     '\nBraking & Traction')[0].split('\n')[2:-2:2]
    # braking_features = df['Features'][0].split('\n\nBraking & Traction')[
    #     1].split('\nLocks & Security')[0].split('\n')[2:-2:2]
    # locks_features = df['Features'][0].split('\n\nLocks & Security')[1].split(
    #     '\nComfort & Convenience')[0].split('\n')[2:-2:2]
    # comfort_features = df['Features'][0].split('\n\nComfort & Convenience')[
    #     1].split('\nTelematics')[0].split('\n')[2:-2:2]
    # telematics_features = df['Features'][0].split('\n\nTelematics')[1].split(
    #     '\nSeats & Upholstery')[0].split('\n')[2:-2:2]
    # seats_features = df['Features'][0].split('\n\nSeats & Upholstery')[
    #     1].split('\nStorage')[0].split('\n')[2:-2:2]
    # storage_features = df['Features'][0].split('\n\nStorage')[1].split(
    #     '\nDoors, Windows, Mirrors & Wipers')[0].split('\n')[2:-2:2]
    # doors_features = df['Features'][0].split('\n\nDoors, Windows, Mirrors & Wipers')[
    #     1].split('\nExterior')[0].split('\n')[2:-2:2]
    # exterior_features = df['Features'][0].split(
    #     '\n\nExterior')[1].split('\nLighting')[0].split('\n')[2:-2:2]
    # lighting_features = df['Features'][0].split('\n\nLighting')[1].split(
    #     '\nInstrumentation')[0].split('\n')[2:-2:2]
    # instrumentation_features = df['Features'][0].split('\n\nInstrumentation')[1].split(
    #     '\nEntertainment, Information & Communication')[0].split('\n')[2:-2:2]
    # entertainment_features = df['Features'][0].split('\n\nEntertainment, Information & Communication')[
    #     1].split('\nManufacturer Warranty')[0].split('\n')[2:-2:2]
    # warranty_features = df['Features'][0].split('\n\nManufacturer Warranty')[
    #     1].split('\n')[2:-2:2]

    # all_features = safety_features + braking_features + locks_features + comfort_features + telematics_features + seats_features + \
    #     storage_features + doors_features + exterior_features + lighting_features + \
    #     instrumentation_features + entertainment_features + warranty_features

    # for feature in tqdm(all_features):
    #     print(feature)
    #     df[feature] = feature_extract(df, re.escape(feature))
    # df.drop('Features', axis=1, inplace=True)

    # df.to_csv('car_data.csv', index=False)
    df.to_csv('data/output.csv', index=False)

