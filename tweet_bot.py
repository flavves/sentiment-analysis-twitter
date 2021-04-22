# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:09:32 2021

@author: yazılım
"""

import selenium
from  selenium import webdriver
import pyautogui
import time



driver_path = "chromedriver"
driver = webdriver.Chrome(executable_path=driver_path)

driver.get("https://twitter.com/login")
#üye girisi

driver.find_element_by_xpath('//*[@id="react-root"]/div/div/div[2]/main/div/div/div[2]/form/div/div[1]/label/div/div[2]/div/input').send_keys("mail")
driver.find_element_by_xpath('//*[@id="react-root"]/div/div/div[2]/main/div/div/div[2]/form/div/div[2]/label/div/div[2]/div/input').send_keys("sifre")
driver.find_element_by_xpath('//*[@id="react-root"]/div/div/div[2]/main/div/div/div[2]/form/div/div[3]/div').click()
#bitti

#veri arama
driver.get("https://twitter.com/search?q=ak%C5%9Fam&src=typed_query&f=live")


time.sleep(4)

enson=driver.find_element_by_xpath('//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[1]/div[2]/nav/div/div[2]/div/div[2]/a')
enson.click()
time.sleep(4)





SCROLL_PAUSE_TIME = 2
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)


    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")

    # break condition
    if new_height == last_height:
        
        break
    last_height = new_height
    a=driver.find_elements_by_xpath('//div[@data-testid="tweet"]')
    for i in a:
        try:
            file =open("yeni3.txt", "a")
            i=i.text
            file.write(i)
            file.close()
            print(i)
        except UnicodeEncodeError:
            continue
        


        
        
#driver.quit()




