# include <esp_now.h>
# include <WiFi.h>
# include <Adafruit_MPU6050.h>
# include <Wire.h>
# include <vector>
# include <algorithm>
# define LED 4
# define PUSH_BUTTON 23
Adafruit_MPU6050 m_p_u;
esp_now_peer_info_t peerInfo;
uint8_t broadcastAddress[] = {0xD0, 0xEF, 0x76, 0x32, 0x56, 0x48}; //MAC Address

typedef struct struct_message {
  char c;
  char b[3];
} struct_message;
struct_message myData;

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status){
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
}

int button = LOW;
double x_COUNT=0;
double y_COUNT=0;
double z_COUNT=0;
double x1,y11,z1;
double z1_low;
std::vector<double> vec_ax;
std::vector<double> vec_ay;
std::vector<double> vec_az;
int i=100;
int j=100;
int count=0;
int new_data;

double findMedian(std::vector<double> arr){
  std::sort(arr.begin(), arr.end());  // Sort the vector
  if(arr[50]<0){
    return arr[10];
  }
  return arr[90];
}
double findlowerMedian(std::vector<double> arr){
  //Return the value received in middle
  return arr[50];
}
//Send data using ESP_NOW protocol
void sent_data(String S, String int_value){
  myData.c=S[0];
  strcpy(myData.b,int_value.c_str());
  Serial.print(myData.c);
  Serial.print(" ");
  Serial.print(myData.b);
  Serial.print('\t');
  esp_err_t result = esp_now_send(broadcastAddress, (uint8_t *) &myData, sizeof(myData));
  if (result == ESP_OK){
    Serial.println("Sending confirmed");
  }
  else{
    Serial.println("Sending error");
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED,OUTPUT);
  pinMode(PUSH_BUTTON,INPUT);
  Wire.begin(21,22); //(SDA_PIN,SCL_PIN)
  if(!m_p_u.begin()){
    Serial.println("Failed to find MPU6050 chip");
    while(1){
      Serial.print(".");
      delay(20);
    }
  }
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  esp_now_register_send_cb(OnDataSent);
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Failed to add peer");
    return;
  }
}

void loop() {
  vec_ax.clear();
  vec_ay.clear();
  vec_az.clear();
  //Calibration
  while(--i>=0){
    sensors_event_t acc,gcc,temp;
    m_p_u.getEvent(&acc,&gcc,&temp);
    digitalWrite(LED,HIGH);
    x_COUNT+=acc.acceleration.x;
    y_COUNT+=acc.acceleration.y;
    z_COUNT+=acc.acceleration.z;
    Serial.print("Calibration- ");
    Serial.println(i);
    delay(100);
    if(i==0){
      Serial.print("Error in x:");
      Serial.println(x_COUNT/100);
      Serial.print("Error in y:");
      Serial.println(y_COUNT/100);
      Serial.print("Error in z:");
      Serial.println(z_COUNT/100+9.8);
      digitalWrite(LED,LOW);
    }
  }
  button = digitalRead(PUSH_BUTTON);
  //Take values when button is pressed
  if(button==HIGH){
    while(--j>=0){
      //LED as indicator
      digitalWrite(LED,HIGH);
      delay(5);
      sensors_event_t acc,gcc,temp;
      m_p_u.getEvent(&acc,&gcc,&temp);
      x1=acc.acceleration.x-x_COUNT/100;
      vec_ax.push_back(x1);
      y11=acc.acceleration.y-y_COUNT/100;
      vec_ay.push_back(y11);
      z1=acc.acceleration.z-z_COUNT/100+9.8;
      vec_az.push_back(z1);
      Serial.print("Data Receiving- ");
      Serial.println(j);
      if(j==0){
        digitalWrite(LED,LOW);
      }
      delay(5);
      button=LOW;
    }

    x1=findMedian(vec_ax);
    y11=findMedian(vec_ay);
    z1=findMedian(vec_az);
    z1_low=findlowerMedian(vec_az);
    j=100;
    count++;
    Serial.print(count);
    Serial.print(". ");
    Serial.print("Acc(m/s2):");
    Serial.print(" (X:"); Serial.print(x1);
    Serial.print(" ,Y:"); Serial.print(y11);
    Serial.print(" ,Z:"); Serial.print(z1);
    Serial.print(" Lower Z: "); Serial.print(z1_low); Serial.print(")");
    Serial.println();
    new_data=1;
    delay(500);
  }

  if (new_data==1){
    new_data=0;
    int speed1=x1*255/10;
    if(speed1>255){
      speed1=255;
    }
    if(speed1<255){
      speed1=-255;
    }
    if(speed1<0){
      speed1=-speed1;
    }

    int speed2=y11*255/10;
    if(speed2>255){
      speed2=255;
    }
    if(speed2<-255){
      speed2=-255;
    }
    if(speed2<0){
      speed2=-speed2;
    }

    int speed3=z1*255/10;
    if(speed3>255){
      speed3=255;
    }
    if(speed3<-255){
      speed3=-255;
    }
    if(speed3<0){
      speed3=-speed3;
    }
    if(-x1>5 && (y11<4 && -y11<4)){
      sent_data("b ",String(speed1));
      Serial.print(speed1);
      Serial.print(" backward motion ");
    }
    else if(x1>5 && y11<4 && -y11<4){
      sent_data("f ",String(speed1));
      Serial.print(speed1);
      Serial.print(" forward motion ");
    }
    else if(-y11>5 && (x1<4 && -x1<4)){
      sent_data("l ",String(speed2));
      Serial.print(speed2);
      Serial.print(" left motion ");
    }
    else if(y11>5 && (x1<4 && -x1<4)){
      sent_data("r ",String(speed2));
      Serial.print(speed2);
      Serial.print(" right motion ");
    }
    else if(-y11>4 && x1>4){
      int speed=(y11>x1)?speed2:speed1;
      sent_data("1 ",String(speed));
      Serial.print(speed);
      Serial.print(" backward left motion ");
    }
    else if(-y11>4&& -x1>4){
      int speed=(y11>-x1)?speed2:speed1;
      sent_data("7 ",String(speed));
      Serial.print(speed);
      Serial.print(" Forward-left motion ");
    }
    else if(y11>4 && -x1>4){
      int speed=(-y11>x1)?speed2:speed1;
      sent_data("3 ",String(speed));
      Serial.print(speed);
      Serial.print(" Backward-right motion ");
    }
    else if(y11>4 && x1>4){
      int speed=(-y11>-x1)?speed2:speed1;
      sent_data("9 ",String(speed));
      Serial.print(speed);
      Serial.print(" forward right motion ");
    }
    else if( z1_low<0){
      sent_data("a ",String(speed3));
      Serial.print(speed3);
      Serial.print(" Anti-clockwise motion ");
    }
    else if( z1>11.8){
      sent_data("c ",String(speed3));
      Serial.print(speed3);
      Serial.print(" Clockwise motion ");
    }
    else{
      int speed=0;
      sent_data("5 ",String(speed));
      Serial.print(speed);
      Serial.print(" Rest ");
    }
  }
}
