#include <Arduino.h>
#include <math.h>
#include <policy_pesos.h>
#include <sensores_voo.h>

// Pinos para os 5 LEDs (Simulando os motores q1 a q5)
const int pinosLED[5] = {13, 19, 14, 27, 26}; 

// Configurações do PWM do ESP32 (LEDC) - API v3.x
const int freqPWM = 5000;
const int resolucaoPWM = 8; // 8 bits: 0 a 255

// Handles das Tasks do FreeRTOS
TaskHandle_t TaskIA_Handle;
TaskHandle_t TaskMotores_Handle;

// Variáveis Globais de Comunicação entre as Tasks
volatile float target_action[5] = {0.0, 0.0, 0.0, 0.0, 0.0}; // Saída bruta da IA (50Hz)
float posicao_suave[5]          = {0.0, 0.0, 0.0, 0.0, 0.0}; // Saída filtrada e suave (250Hz)

// Cálculo explícito do Alpha para o Filtro Passa-Baixa
const float freq_corte = 7.0;         // 7Hz ditado pela aerodinâmica/inércia
const float dt_motores = 1.0 / 250.0; // 0.004 segundos (250Hz)
const float alpha_servo = 1.0 - exp(-2.0 * PI * freq_corte * dt_motores);

const int NUM_AMOSTRAS_TESTE = 100;
int indice_amostra_atual = 0;

// Memória da última ação permitida e o delta máximo calculado (Rate Limiter)
float last_policy_action[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
const float limiter_deltas[5] = {0.26, 0.33428571, 0.26, 0.33428571, 0.2925}; // Ajuste com os valores do seu Python

// O arquivo policy_pesos.h exportado pelo Python trará estas exatas definições:
// const float mlp_extractor_policy_net_0_weight[256][540] = {...};
// const float mlp_extractor_policy_net_0_bias[256] = {...};
// const float mlp_extractor_policy_net_2_weight[256][256] = {...};
// const float mlp_extractor_policy_net_2_bias[256] = {...};
// const float action_net_weight[5][256] = {...};
// const float action_net_bias[5] = {...};

// A matriz de sensores exportada (Sensores Voo)
// const float sensores_voo[100][540] = {...};


// ---------------------------------------------------------
// TASK 1: INFERÊNCIA DA IA (O "Cérebro" a 50Hz / 20ms)
// ---------------------------------------------------------

void inferir_acao(const float* obs_atual, float* saida_motores) {
  float ativacao_camada1[256];
  float ativacao_camada2[256];

  // ---------------------------------------------------------
  // CAMADA OCULTA 1: 540 Entradas -> 256 Neurônios
  // ---------------------------------------------------------
  for (int i = 0; i < 256; i++) {
    float soma = mlp_extractor_policy_net_0_bias[i];
    for (int j = 0; j < 540; j++) {
      soma += mlp_extractor_policy_net_0_weight[i][j] * obs_atual[j];
    }
    ativacao_camada1[i] = tanh(soma);
  }

  // ---------------------------------------------------------
  // CAMADA OCULTA 2: 256 Entradas -> 256 Neurônios
  // ---------------------------------------------------------
  for (int i = 0; i < 256; i++) {
    float soma = mlp_extractor_policy_net_2_bias[i];
    for (int j = 0; j < 256; j++) {
      soma += mlp_extractor_policy_net_2_weight[i][j] * ativacao_camada1[j];
    }
    ativacao_camada2[i] = tanh(soma);
  }

  // ---------------------------------------------------------
  // CAMADA DE SAÍDA: 256 Entradas -> 5 Motores
  // ---------------------------------------------------------
  for (int i = 0; i < 5; i++) {
    float soma = action_net_bias[i];
    for (int j = 0; j < 256; j++) {
      soma += action_net_weight[i][j] * ativacao_camada2[j];
    }
    // O limite [-1, 1] é aplicado EXCLUSIVAMENTE AQUI para as 5 saídas finais
    saida_motores[i] = constrain(soma, -1.0f, 1.0f);
  }
}

void TaskInferenciaIA(void *pvParameters) {
  const TickType_t xFrequency = 20 / portTICK_PERIOD_MS;
  TickType_t xLastWakeTime = xTaskGetTickCount();

  for (;;) {
    const float* leitura_frame_atual = sensores_voo[indice_amostra_atual];
    
    // 1. Array temporário para receber a ação pura da rede neural
    float acao_bruta[5];
    inferir_acao(leitura_frame_atual, acao_bruta);

    // 2. Aplica o Rate Limiter idêntico ao do ambiente Python
    for(int i = 0; i < 5; i++) {
      float limite_inf = last_policy_action[i] - limiter_deltas[i];
      float limite_sup = last_policy_action[i] + limiter_deltas[i];
      
      // Clipa a ação bruta dentro da janela de variação permitida
      float acao_limitada = constrain(acao_bruta[i], limite_inf, limite_sup);
      
      // Atualiza a memória e a caixa de correio global (target_action)
      last_policy_action[i] = acao_limitada;
      target_action[i]      = acao_limitada;
    }
    
    // CORREÇÃO 1: Impede que o array leia lixo de memória
    indice_amostra_atual++;
    if (indice_amostra_atual >= NUM_AMOSTRAS_TESTE) {
      indice_amostra_atual = 0;
    }

    // CORREÇÃO 2: Devolve o fôlego para o processador e crava a taxa em 50Hz
    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}

// ---------------------------------------------------------
// TASK 2: FILTRO E ATUADORES (O "Músculo" a 250Hz / 4ms)
// ---------------------------------------------------------
void TaskControleMotores(void *pvParameters) {
  const TickType_t xFrequency = 4 / portTICK_PERIOD_MS;
  TickType_t xLastWakeTime = xTaskGetTickCount();

  for (;;) {
    for(int i = 0; i < 5; i++) {
      // 1. Aplica o Filtro Passa-Baixa na leitura mais recente da IA
      posicao_suave[i] = (alpha_servo * target_action[i]) + ((1.0 - alpha_servo) * posicao_suave[i]);
      
      // 2. Limita e converte para PWM
      float acao_clipada = constrain(posicao_suave[i], -1.0, 1.0); 
      int pwm_val = (acao_clipada + 1.0) * 127.5; 
      
      // 3. Envia o pulso elétrico para o LED/Servo
      ledcWrite(pinosLED[i], pwm_val);
    }

    // O Plotter agora mostra a curva suave fluindo a 250Hz
    Serial.print(posicao_suave[0]); Serial.print(",");
    Serial.print(posicao_suave[1]); Serial.print(",");
    Serial.print(posicao_suave[2]); Serial.print(",");
    Serial.print(posicao_suave[3]); Serial.print(",");
    Serial.println(posicao_suave[4]); 

    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}

// ---------------------------------------------------------
// SETUP E INICIALIZAÇÃO
// ---------------------------------------------------------
void setup() {
  Serial.begin(115200);

  for(int i = 0; i < 5; i++) {
    ledcAttach(pinosLED[i], freqPWM, resolucaoPWM);
  }

  // Task da IA (Prioridade 1 - Baixa)
  xTaskCreatePinnedToCore(TaskInferenciaIA, "Inferencia_IA", 10000, NULL, 1, &TaskIA_Handle, 1);
  
  // Task dos Motores (Prioridade 2 - Alta)
  // O Músculo sempre tem prioridade sobre o Cérebro para evitar engasgos nos motores físicos
  xTaskCreatePinnedToCore(TaskControleMotores, "Controle_Motores", 5000, NULL, 2, &TaskMotores_Handle, 1);
}

void loop() {
  vTaskDelay(1000 / portTICK_PERIOD_MS); 
}