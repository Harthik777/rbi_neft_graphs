import os
import time
import json
import requests
import random
import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

user_agent = "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
base_url = "https://m.rbi.org.in/scripts/NEFTView.aspx"
root = "RBI_Data"
#sheets = ["NEFT", "RTGS", "Mobile_Banking", "Internet_Banking"]

os.makedirs(root, exist_ok=True)
root_path = os.path.join(os.getcwd(),root)

min_year = 2016
current_year = datetime.datetime.now().year

options = Options()
options.add_argument(user_agent)
options.headless = False

driver = webdriver.Chrome(options=options)
driver.get(base_url)

cookies = driver.get_cookies()

with open(os.path.join(root, "cookies.json"), "w+") as cookie_file:
    json.dump(cookies, cookie_file)

driver.refresh()

for cookie in cookies:
    driver.add_cookie(cookie)

driver.refresh()

print("Please solve the captcha manually in the browser.")
time.sleep(15)

for year in range(min_year, current_year + 1):

    os.chdir(root_path)
    os.makedirs(str(year),exist_ok=True)

    current_path = os.path.join(root_path,str(year))
    os.chdir(current_path)


    try:
        button = driver.find_element(By.ID, f"btn{year}")
        button.click()
        time.sleep(random.uniform(0.5, 1.5))

        all_months = driver.find_element(By.ID, f"{year}0")
        all_months.click()
        time.sleep(random.uniform(0.5, 1.5))

        soup = BeautifulSoup(driver.page_source, "html.parser")

        table = soup.find("tbody")

        for tag in table.contents:
            b_tag = tag.find("b")
            link = tag.find("td",nowrap="")
            #print(tag)
            if(b_tag):
                month = b_tag.text.split("-")[0]
            elif(link):
                xlsx_link = link.find("a").get("href")
                print(xlsx_link.split("."))

                file_content = requests.get(xlsx_link).content
                file_path = os.path.join(os.getcwd(),month.rstrip()+"."+(xlsx_link.split(".")[-1]))
                print(file_path)

                with open(file_path,'wb+') as file:
                    file.write(file_content)

                print("NEXT")
            else:
                continue
        time.sleep(random.uniform(2.5, 3.5))


    except Exception as e:
        print(f"Skipping {year} due to error: {str(e)}")

print("Download Completed.")
driver.quit()
