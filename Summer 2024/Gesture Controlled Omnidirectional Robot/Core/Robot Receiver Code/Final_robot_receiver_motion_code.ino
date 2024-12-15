//Connect the pins as mentioned
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
# include <esp_now.h>
# include <WiFi.h>
int count=0;
//Storing the received data
typedef struct struct_message {
  char c;
  char b[3];
} struct_message;
struct_message myData;

void OnDataRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  memcpy(&myData, data, sizeof(myData));
  count=1;
}

void Forward(int receivedspeed){
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
void Backward(int receivedspeed){
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
void Leftward(int receivedspeed){
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
void Rightward(int receivedspeed){
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
void Anticlockwise(int receivedspeed){
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
void Clockwise(int receivedspeed){
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
void Forward_left(int receivedspeed){
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
void Forward_right(int receivedspeed){
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
void Backward_left(int receivedspeed){
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
void Backward_right(int receivedspeed){
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
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  esp_now_register_recv_cb(OnDataRecv);

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
  //D0:EF:76:32:56:48 Robot MAC address
  if(count==1){
    count=0;
    int receivedspeed = atoi(myData.b); //Receive data in array of char format and convert to integer
    char c=myData.c;
    Serial.print(c);
    Serial.print(" ");
    Serial.print(receivedspeed); // Define speed between 0 to 255 with Minimum at 0 and Maximum at 255 proportionally
    Serial.print('\t');

    if(c=='f'){
      Forward(receivedspeed);
    }
    else if(c=='b'){
      Backward(receivedspeed);
    }
    else if(c=='l'){
      Leftward(receivedspeed);
    }
    else if(c=='r'){
      Rightward(receivedspeed);
    }
    else if(c=='a'){
      Anticlockwise(receivedspeed);
    }
    else if(c=='c'){
      Clockwise(receivedspeed);
    }
    else if(c=='7'){
      Forward_left(receivedspeed);
    }
    else if(c=='9'){
      Forward_right(receivedspeed);
    }
    else if(c=='1'){
      Backward_left(receivedspeed);
    }
    else if(c=='3'){
      Backward_right(receivedspeed);
    }
    else{
      Stop();
    }
  }
}
