from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
driver = webdriver.Chrome()


def getKq(textTime):
    # driver.delete_all_cookies()
    # driver.refresh()
    driver.get(f"https://www.minhngoc.net.vn/ket-qua-xo-so/mien-bac/{textTime}.html")
    driver.implicitly_wait(2)
    kqs = []
    table_0 = driver.find_elements(By.CLASS_NAME, 'box_kqxs')[0]
    elements = table_0.find_elements(By.CLASS_NAME, 'giaiSo')
    for element in elements:
        kqs.append(element.text)
    # driver.quit()
    return kqs

