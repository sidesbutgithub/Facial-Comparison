from selenium.webdriver.common.by import By
from selenium import webdriver

url = 'https://stackoverflow.com/questions/3422262/how-to-take-a-screenshot-with-selenium-webdriver'
path = 'scrape.jpg'

driver = webdriver.Firefox()
driver.get(url)
driver.save_full_page_screenshot(path)

driver.quit()