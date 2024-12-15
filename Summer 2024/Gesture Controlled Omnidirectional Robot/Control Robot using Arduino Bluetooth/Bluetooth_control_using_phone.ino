//Send "f" for forward
//Send "b" for backward
//Send "r" for rightward
//Send "l" for leftward
//Send "c" for clockwise
//Send "a" for anticlockwise
//Send "1" for backward left
//Send "3" for backward right
//Send "7" for forward left
//Send "9" for forward right
//Connent bluetooth of Arduino BlueControl App with robot receiver esp32 and sent data in Terminal of that App.

# define M1_PWM 4
# define M1_IN1 22
# define M1_IN2 23
# define M2_PWM 15
# define M2_IN1 19
# define M2_IN2 18
# define M3_PWM 12
# define M3_IN1 32
# define M3_IN2 33
# define M4_PWM 13
# define M4_IN1 25
# define M4_IN2 26
# include <BluetoothSerial.h>

BluetoothSerial SerialBT;
int receivedspeed=255;
void Forward(){
  Serial.println("Forward Motion");
  analogWrite(M1_PWM, receivedspeed);
  digitalWrite(M1_IN1, HIGH);
  digitalWrite(M1_IN2, LOW);

  analogWrite(M2_PWM, receivedspeed);
  digitalWrite(M2_IN1, HIGH);
  digitalWrite(M2_IN2, LOW);

  analogWrite(M3_PWM, receivedspeed);
  digitalWrite(M3_IN1, HIGH);
  digitalWrite(M3_IN2, LOW);

  analogWrite(M4_PWM, receivedspeed);
  digitalWrite(M4_IN1, HIGH);
  digitalWrite(M4_IN2, LOW);
}
void Backward(){
  Serial.println("Backward Motion");
  analogWrite(M1_PWM, receivedspeed);
  digitalWrite(M1_IN1, LOW);
  digitalWrite(M1_IN2, HIGH);

  analogWrite(M2_PWM, receivedspeed);
  digitalWrite(M2_IN1, LOW);
  digitalWrite(M2_IN2, HIGH);

  analogWrite(M3_PWM, receivedspeed);
  digitalWrite(M3_IN1, LOW);
  digitalWrite(M3_IN2, HIGH);

  analogWrite(M4_PWM, receivedspeed);
  digitalWrite(M4_IN1, LOW);
  digitalWrite(M4_IN2, HIGH);
}
void Leftward(){
  Serial.println("Leftward Motion");
  analogWrite(M1_PWM, receivedspeed);
  digitalWrite(M1_IN1, LOW);
  digitalWrite(M1_IN2, HIGH);

  analogWrite(M2_PWM, receivedspeed);
  digitalWrite(M2_IN1, HIGH);
  digitalWrite(M2_IN2, LOW);

  analogWrite(M3_PWM, receivedspeed);
  digitalWrite(M3_IN1, LOW);
  digitalWrite(M3_IN2, HIGH);

  analogWrite(M4_PWM, receivedspeed);
  digitalWrite(M4_IN1, HIGH);
  digitalWrite(M4_IN2, LOW);
}
void Rightward(){
  Serial.println("Rightward Motion");
  analogWrite(M1_PWM, receivedspeed);
  digitalWrite(M1_IN1, HIGH);
  digitalWrite(M1_IN2, LOW);

  analogWrite(M2_PWM, receivedspeed);
  digitalWrite(M2_IN1, LOW);
  digitalWrite(M2_IN2, HIGH);

  analogWrite(M3_PWM, receivedspeed);
  digitalWrite(M3_IN1, HIGH);
  digitalWrite(M3_IN2, LOW);

  analogWrite(M4_PWM, receivedspeed);
  digitalWrite(M4_IN1, LOW);
  digitalWrite(M4_IN2, HIGH);
}
void Anticlockwise(){
  Serial.println("Anticlockwise Motion");
  analogWrite(M1_PWM, receivedspeed);
  digitalWrite(M1_IN1, LOW);
  digitalWrite(M1_IN2, HIGH);

  analogWrite(M2_PWM, receivedspeed);
  digitalWrite(M2_IN1, HIGH);
  digitalWrite(M2_IN2, LOW);

  analogWrite(M3_PWM, receivedspeed);
  digitalWrite(M3_IN1, HIGH);
  digitalWrite(M3_IN2, LOW);

  analogWrite(M4_PWM, receivedspeed);
  digitalWrite(M4_IN1, LOW);
  digitalWrite(M4_IN2, HIGH);
}
void Clockwise(){
  Serial.println("Clockwise Motion");
  analogWrite(M1_PWM, receivedspeed);
  digitalWrite(M1_IN1, HIGH);
  digitalWrite(M1_IN2, LOW);

  analogWrite(M2_PWM, receivedspeed);
  digitalWrite(M2_IN1, LOW);
  digitalWrite(M2_IN2, HIGH);

  analogWrite(M3_PWM, receivedspeed);
  digitalWrite(M3_IN1, LOW);
  digitalWrite(M3_IN2, HIGH);

  analogWrite(M4_PWM, receivedspeed);
  digitalWrite(M4_IN1, HIGH);
  digitalWrite(M4_IN2, LOW);
}
void Forward_left(){
  Serial.println("Forward_left");
  analogWrite(M1_PWM, receivedspeed);
  digitalWrite(M1_IN1, LOW);
  digitalWrite(M1_IN2, LOW);

  analogWrite(M2_PWM, receivedspeed);
  digitalWrite(M2_IN1, HIGH);
  digitalWrite(M2_IN2, LOW);

  analogWrite(M3_PWM, receivedspeed);
  digitalWrite(M3_IN1, LOW);
  digitalWrite(M3_IN2, LOW);

  analogWrite(M4_PWM, receivedspeed);
  digitalWrite(M4_IN1, HIGH);
  digitalWrite(M4_IN2, LOW);
}
void Forward_right(){
  Serial.println("Forward_right");
  analogWrite(M1_PWM, receivedspeed);
  digitalWrite(M1_IN1, HIGH);
  digitalWrite(M1_IN2, LOW);

  analogWrite(M2_PWM, receivedspeed);
  digitalWrite(M2_IN1, LOW);
  digitalWrite(M2_IN2, LOW);

  analogWrite(M3_PWM, receivedspeed);
  digitalWrite(M3_IN1, HIGH);
  digitalWrite(M3_IN2, LOW);

  analogWrite(M4_PWM, receivedspeed);
  digitalWrite(M4_IN1, LOW);
  digitalWrite(M4_IN2, LOW);
}
void Backward_left(){
  Serial.println("Backward_left");
  analogWrite(M1_PWM, receivedspeed);
  digitalWrite(M1_IN1, LOW);
  digitalWrite(M1_IN2, HIGH);

  analogWrite(M2_PWM, receivedspeed);
  digitalWrite(M2_IN1, LOW);
  digitalWrite(M2_IN2, LOW);

  analogWrite(M3_PWM, receivedspeed);
  digitalWrite(M3_IN1, LOW);
  digitalWrite(M3_IN2, HIGH);

  analogWrite(M4_PWM, receivedspeed);
  digitalWrite(M4_IN1, LOW);
  digitalWrite(M4_IN2, LOW);
}
void Backward_right(){
  Serial.println("Backward_right");
  analogWrite(M1_PWM, receivedspeed);
  digitalWrite(M1_IN1, LOW);
  digitalWrite(M1_IN2, LOW);

  analogWrite(M2_PWM, receivedspeed);
  digitalWrite(M2_IN1, LOW);
  digitalWrite(M2_IN2, HIGH);

  analogWrite(M3_PWM, receivedspeed);
  digitalWrite(M3_IN1, LOW);
  digitalWrite(M3_IN2, LOW);

  analogWrite(M4_PWM, receivedspeed);
  digitalWrite(M4_IN1, LOW);
  digitalWrite(M4_IN2, HIGH);
}
void Stop(){
  Serial.println("Stop");
  analogWrite(M1_PWM, 0);
  digitalWrite(M1_IN1, LOW);
  digitalWrite(M1_IN2, LOW);

  analogWrite(M2_PWM, 0);
  digitalWrite(M2_IN1, LOW);
  digitalWrite(M2_IN2, LOW);

  analogWrite(M3_PWM, 0);
  digitalWrite(M3_IN1, LOW);
  digitalWrite(M3_IN2, LOW);

  analogWrite(M4_PWM, 0);
  digitalWrite(M4_IN1, LOW);
  digitalWrite(M4_IN2, LOW);
}
void setup() {
  Serial.begin(115200);
  SerialBT.begin("Omni_directional_robot");
  Serial.println("Bluetooth initialized. Ready to pair.");
  pinMode(M1_PWM, OUTPUT);
  pinMode(M1_IN1, OUTPUT);
  pinMode(M1_IN2, OUTPUT);
  pinMode(M2_PWM, OUTPUT);
  pinMode(M2_IN1, OUTPUT);
  pinMode(M2_IN2, OUTPUT);
  pinMode(M3_PWM, OUTPUT);
  pinMode(M3_IN1, OUTPUT);
  pinMode(M3_IN2, OUTPUT);
  pinMode(M4_PWM, OUTPUT);
  pinMode(M4_IN1, OUTPUT);
  pinMode(M4_IN2, OUTPUT);
}

void loop() {
  if(SerialBT.available()){
    char c= char(SerialBT.read()); //Read data
    Serial.print("Received: ");
    Serial.println(c);
    //For Forward Motion
    if(c=='f'){
      Forward();
    }
    //For Backward Motion
    else if(c=='b'){
      Backward();
    }
    //For Leftward Motion
    else if(c=='l'){
      Leftward();
    }
    //For Rightward Motion
    else if(c=='r'){
      Rightward();
    }
    //For Anti-Clockwise Motion
    else if(c=='a'){
      Anticlockwise();
    }
    //For Clockwise Motion
    else if(c=='c'){
      Clockwise();
    }
    //For Diagonal Motion
    else if(c=='7'){
      Forward_left();
    }
    else if(c=='9'){
      Forward_right();
    }
    else if(c=='1'){
      Backward_left();
    }
    else if(c=='3'){
      Backward_right();
    }
    else{
      Stop();
    }
  }
}
