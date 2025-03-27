/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body for TinyML Image Recognition
  * @author         : MEng Design Project - Edge AI Sustainability
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h" // Your converted TensorFlow Lite model

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
#define TENSOR_ARENA_SIZE 16384 // 16KB tensor allocation space
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// Power measurement tracking
volatile uint32_t inference_count = 0;
volatile float total_energy_consumed = 0.0f;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_ADC1_Init(void);
static void TinyML_Init(void);
static void Measure_Power(void);

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */
  HAL_Init();
  SystemClock_Config();
  
  /* Initialize peripherals */
  MX_GPIO_Init();
  MX_ADC1_Init(); // For power measurement
  
  /* Initialize TinyML Inference */
  TinyML_Init();

  /* USER CODE END 1 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* Perform Inference */
    Perform_Inference();
    
    /* Measure Power Consumption */
    Measure_Power();
    
    /* Optional: Log Results */
    Log_Inference_Results();
    
    HAL_Delay(1000); // 1-second delay between inferences
    
    /* USER CODE BEGIN WHILE */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 100;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  HAL_RCC_OscConfig(&RCC_OscInitStruct);

  /** Initializes the CPU, AHB and APB buses clocks */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3);
}

/**
  * @brief TinyML Initialization Function
  * @retval None
  */
static void TinyML_Init(void)
{
  // Initialize TensorFlow Lite Micro
  static tflite::MicroErrorReporter micro_error_reporter;
  static tflite::AllOpsResolver resolver;

  // Create interpreter
  static const tflite::Model* model = tflite::GetModel(g_model);
  interpreter = new tflite::MicroInterpreter(
    model, 
    resolver, 
    tensor_arena, 
    TENSOR_ARENA_SIZE, 
    &micro_error_reporter
  );

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    // Handle allocation error
    while(1); // Halt if initialization fails
  }

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

/**
  * @brief Power Measurement Function
  * @retval None
  */
static void Measure_Power(void)
{
  // Configure ADC for current/voltage measurement
  uint16_t adc_value;
  HAL_ADC_Start(&hadc1);
  HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY);
  adc_value = HAL_ADC_GetValue(&hadc1);
  
  // Convert ADC value to power
  float voltage = (adc_value * 3.3f) / 4096.0f;
  float current = voltage / 100.0f; // Assuming a 100Ω shunt resistor
  float power = voltage * current;
  
  // Accumulate energy consumption
  total_energy_consumed += power * 0.001f; // Convert to mWh
  inference_count++;
}

/**
  * @brief Inference Performance Logging
  * @retval None
  */
static void Log_Inference_Results(void)
{
  // Output results via UART or store in memory
  printf("Inference #%lu\n", inference_count);
  printf("Average Energy per Inference: %.4f mWh\n", 
         total_energy_consumed / inference_count);
  
  // Optional: Log model output/classification results
  for (int i = 0; i < output->dims->size; i++) {
    printf("Output %d: %f\n", i, output->data.f[i]);
  }
}

/**
  * @brief Perform Machine Learning Inference
  * @retval None
  */
static void Perform_Inference(void)
{
  // Prepare input data (assumption: preprocessed image data)
  // This would typically involve camera input or preset test data
  for (int i = 0; i < input->dims->size; i++) {
    input->data.f[i] = /* your preprocessed input data */;
  }

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    // Handle inference error
    while(1);
  }
}
