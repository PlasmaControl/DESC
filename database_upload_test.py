# flake8: noqa
import os
import zipfile

from database_converter import desc_to_csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def get_driver():
    # List of supported browsers
    browsers = ["firefox", "chrome", "edge", "safari"]

    for browser in browsers:
        try:
            if browser == "firefox":
                options = webdriver.FirefoxOptions()
                options.headless = True
                return webdriver.Firefox(options=options, executable_path="geckodriver")

            elif browser == "chrome":
                options = webdriver.ChromeOptions()
                options.headless = True
                return webdriver.Chrome(options=options, executable_path="chromedriver")

            elif browser == "edge":
                options = webdriver.EdgeOptions()
                options.use_chromium = True
                options.add_argument("headless")
                return webdriver.Edge(options=options)

            elif browser == "safari":
                return webdriver.Safari()

        except Exception as e:
            print(f"Failed to initialize {browser} webdriver:", e)

    # If no browser was successfully initialized, return None
    return None


def zip_and_upload(
    filename, configid=None, initialization_method="surface", user="yge"
):
    zip_upload_button_id = "zipToUpload"
    csv_upload_button_id = "descToUpload"
    cfg_upload_button_id = "configToUpload"
    confirm_button_id = "confirmDesc"

    # Zip the files
    zip_filename = filename + ".zip"
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        zipf.write(filename + ".h5")
        if os.path.exists(filename + "_input.txt"):
            zipf.write(filename + "_input.txt")

    csv_filename = "desc_runs.csv"
    config_csv_filename = "configurations.csv"
    if os.path.exists(csv_filename):
        os.remove(csv_filename)
        print(f"Previous {csv_filename} has been deleted.")
    if os.path.exists(config_csv_filename):
        os.remove(config_csv_filename)
        print(f"Previous {config_csv_filename} has been deleted.\n")

    print("Creating desc_runs.csv and configurations.csv...\n")
    desc_to_csv(
        filename + ".h5",  # output filename
        name=configid,  # some string descriptive name, not necessarily unique
        provenance="Test upload zip",
        inputfilename=None,
        current=True,
        deviceid=None,
        config_class=None,
        user_updated=user,
        user_created=user,
        initialization_method=initialization_method,
    )

    driver = get_driver()
    driver.get("https://ye2698.mycpanel.princeton.edu/import-page/")

    try:
        # Upload the zip file
        file_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, zip_upload_button_id))
        )
        file_input.send_keys(os.path.abspath(zip_filename))

        # Upload the csv file for desc_runs
        file_input2 = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, csv_upload_button_id))
        )
        file_input2.send_keys(os.path.abspath(csv_filename))

        # Upload the csv file for configurations
        file_input3 = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, cfg_upload_button_id))
        )
        file_input3.send_keys(os.path.abspath(config_csv_filename))

        # Confirm the upload
        confirm_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, confirm_button_id))
        )
        confirm_button.click()

        # Wait for the messageContainer div to contain text
        WebDriverWait(driver, 10).until(
            lambda driver: driver.find_element(By.ID, "messageContainer").text.strip()
            != ""
        )

        # Extract and print the message
        message_element = driver.find_element(By.ID, "messageContainer")
        message = message_element.text
        print(message)
    except:
        # Extract and print the message
        message_element = driver.find_element(By.ID, "messageContainer")
        message = message_element.text
        print(message)

    finally:
        # Clean up resources
        driver.quit()
        os.remove(zip_filename)


# Example usage:
configid = "precise_QA"
user = "yge"
zip_and_upload(
    "precise_QA_poincare_N3", configid, initialization_method="poincare", user=user
)
