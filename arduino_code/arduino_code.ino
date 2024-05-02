#include <Adafruit_MAX31865.h>
#include <Wire.h>
#include "rgb_lcd.h" // library is located at  https://github.com/Seeed-Studio/Grove_LCD_RGB_Backlight
Adafruit_MAX31865 thermo = Adafruit_MAX31865(10, 11, 12, 13);
#define RREF 430.0
#define RNOMINAL 100.0
bool temperatureEndsWithQuarter = false;
rgb_lcd lcd;
void setup() {
Serial.begin(115200);
//Serial.println("Début de la prise de température");
thermo.begin(MAX31865_3WIRE);
//lcd.begin(16,2);
}
void loop() {
uint16_t rtd = thermo.readRTD();
float temperature = thermo.temperature(RNOMINAL, RREF);
//Serial.println(temperature);
updateTemperatureEndsWithQuarter(temperature);
Serial.println(temperatureEndsWithQuarter);

//lcd.noDisplay(); // turns the display off
  //  delay(400);   // time while display is off and waiting for a temp reading refresh
   // lcd.display();    // home is set by default to value lcd.setCursor(0, 0);
  //  lcd.print("Temp is = ");
  //  lcd.print(thermo.temperature(RNOMINAL, RREF));  // displays the temp float variable as read by the RTD
   // lcd.print((char)223); // print ° character
    //lcd.setCursor(0, 1);  // sets the cursor to the second line of the 16x2 display
  //  lcd.print("         CELCIUS");
   // delay(4000);  // displays temp value for 4 seconds
    //lcd.clear();  // clears the display

uint8_t fault = thermo.readFault();
if (fault) {
Serial.print("Fault 0x"); Serial.println(fault, HEX);
if (fault & MAX31865_FAULT_HIGHTHRESH) {
Serial.println("RTD High Threshold");
}
if (fault & MAX31865_FAULT_LOWTHRESH) {
Serial.println("RTD Low Threshold");
}
if (fault & MAX31865_FAULT_REFINLOW) {
Serial.println("REFIN- > 0.85 x Bias");
}
if (fault & MAX31865_FAULT_REFINHIGH) {
Serial.println("REFIN- < 0.85 x Bias - FORCE- open");
}
if (fault & MAX31865_FAULT_RTDINLOW) {
Serial.println("RTDIN- < 0.85 x Bias - FORCE- open");
}
if (fault & MAX31865_FAULT_OVUV) {
Serial.println("Under/Over voltage");
}
thermo.clearFault();
}
delay(10);
}
void updateTemperatureEndsWithQuarter(float temperature) {
float diff = temperature - int(temperature); // Get the decimal part of the temperature
if (temperature >= 26 && temperature <= 50) {
if ((diff >= 0.00 && diff <= 0.10) ||
(diff >= 0.20 && diff <= 0.30) ||
(diff >= 0.50 && diff <= 0.60) ||
(diff >= 0.70 && diff <= 0.80)) {
temperatureEndsWithQuarter = true;
} else {
temperatureEndsWithQuarter = false;
}
} else {
temperatureEndsWithQuarter = false;
}
}