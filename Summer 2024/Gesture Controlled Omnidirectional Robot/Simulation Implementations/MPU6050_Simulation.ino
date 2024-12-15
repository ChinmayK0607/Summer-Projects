#include<Adafruit_MPU6050.h>
Adafruit_MPU6050 m_p_u;
void setup() {
  Serial.begin(115200);
  while(!Serial)
  delay(20);
  Wire.begin(21,32);
  if(!m_p_u.begin()){
    Serial.println("Failed to find MPU6050 chip");
    while(1){
      delay(20);
    }
  }
}

void loop() {
  sensors_event_t acc,gcc,temp;
  m_p_u.getEvent(&acc,&gcc,&temp); //Get data

  Serial.print("Acc(m/s2):");
  Serial.print(" (X:"); Serial.print(acc.acceleration.x);
  Serial.print(" ,Y:"); Serial.print(acc.acceleration.y);
  Serial.print(" ,Z:"); Serial.print(acc.acceleration.z);
  Serial.print(") | ");

  Serial.print("Temp(°C):"); Serial.print(temp.temperature);
  Serial.print(" | ");

  Serial.print("Gyro(°/s):");
  Serial.print(" (X:"); Serial.print(gcc.gyro.x*180/3.14);
  Serial.print(" ,Y:"); Serial.print(gcc.gyro.y*180/3.14);
  Serial.print(" ,Z:"); Serial.print(gcc.gyro.z*180/3.14);
  Serial.print(")");
  Serial.println();
  delay(500);
}
