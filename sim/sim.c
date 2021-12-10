#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include <mongoc/mongoc.h>
#include <bson/bson.h>

#define WEIGHT_FRAC_BITS 13
#define FACTOR 8192

typedef struct neuron {
  int16_t bias;
  int weight_cnt;
  int16_t *weights;
} neuron_t;

typedef struct layer {
  int neuron_cnt;
  neuron_t **neurons;
} layer_t;

typedef struct net {
  int layer_cnt;
  layer_t **layers;
} net_t;

static void die(char *error) {
  fprintf(stderr, "Error: %s\n", error);
  exit(EXIT_FAILURE);
}

static double to_double(int16_t value) {
  return (double)value / (double)FACTOR;
}

static int16_t read_value(FILE *weight_file) {
  uint8_t bytes[2];
  if (fread(&bytes[0], 1, 2, weight_file) != 2) {
    die("Could not read value");
  }
  return (((int16_t)bytes[0]) << 8) | ((int16_t)bytes[1]);
}

static void print_net(net_t *net) {
  for (int i = 0; i < net->layer_cnt; i++) {
    printf("Layer: %d\n", i);
    for (int j = 0; j < net->layers[i]->neuron_cnt; j++) {
      printf("\tNeuron: %d\n", j);
      printf("\t\tBias: %.6f\n", to_double(net->layers[i]->neurons[j]->bias));
      printf("\t\tWeights: [");
      for (int k = 0; k < net->layers[i]->neurons[j]->weight_cnt; k++) {
        if (k > 0) {
          printf(", ");
        }
        printf("%.6f", to_double(net->layers[i]->neurons[j]->weights[k]));
      }
      printf("]\n");
    }
  }
}

static int net_forward(net_t *net, const uint8_t *sub_image) {
  int16_t *layers_out[net->layer_cnt];
  bool test = true;
  // Create tanh rom array
  uint16_t tanh_rom [32768];
  for (int k = 0; k < 32768; k++) {
    double tanh_x = 0.00003051757 * k;
    uint16_t temp = (uint16_t)(tanh(tanh_x) * 8192);
    tanh_rom[k] = temp;
  }
  for (int i = 0; i < net->layer_cnt; i++) {
    layers_out[i] = (int16_t*)malloc(sizeof(int16_t) * net->layers[i]->neuron_cnt);
    for (int j = 0; j < net->layers[i]->neuron_cnt; j++) {
      int32_t accumulation = (int32_t)net->layers[i]->neurons[j]->bias;
      double accumulation_f = to_double(net->layers[i]->neurons[j]->bias);
      // Loop through # of weights for neuron, equals # of inputs
      for (int k = 0; k < net->layers[i]->neurons[j]->weight_cnt; k++) {
        int16_t weight = net->layers[i]->neurons[j]->weights[k]; // Make sure weights are converted to +/-2.13
        int16_t input = (i == 0) ? (((int16_t)sub_image[k]) << 5) : layers_out[i - 1][k];
        int32_t result = (((int32_t)input) * ((int32_t)weight)) >> 13;
        accumulation += result;
        if (test) {
          double weight_f = to_double(net->layers[i]->neurons[j]->weights[k]); 
          float input_f = (i == 0) ? (((float)sub_image[k])/256.0f) : layers_out[i - 1][k];
          double result_f = ((double)input_f) * ((double)weight_f);
          accumulation_f += result_f;
          //printf("num:%d\nFloating point:%f\nFixed Point: %f\nWeight: %d\nInput: %d\nResult: %d\n\n", k, accumulation_f, (accumulation*0.00012207031), weight, input, result);
        }
      }
      //printf("Floating point 32bit:%f\nFixed Point 16 bit implementation: %f\n", accumulation_f, (accumulation*0.00012207031));
      //exit(0);

      // Round Accumulation for Activation funtion.
      uint16_t rounded_accum;
      bool accum_positive = false;
      if (accumulation >= 0) {
        accum_positive = true;
      }
      uint32_t abs_accumulation = abs(accumulation);
      // abs_accumulation = abs_accumulation >> 7;
      if (abs_accumulation > 32767) {
        rounded_accum = 32767;
      } else {
        rounded_accum = ((uint16_t)abs_accumulation);
      }
      

      if (i != net->layer_cnt - 1) {
        int16_t activation_output;
        activation_output = (int16_t)tanh_rom[rounded_accum];
        if (!accum_positive) {
          activation_output = -(activation_output);
        }
        layers_out[i][j] = activation_output;
      } else {
        int16_t c_output;
        c_output = (int16_t)rounded_accum;
        if (!accum_positive) {
          c_output = -(c_output);
        }
        layers_out[i][j] = c_output;
      }
    }
  }

  // Selects the max C neuron's output
  int max_c_index;
  int16_t temp_max = INT16_MIN;
  for(int neurons = 0; neurons < 36; neurons++) {
    int16_t temp = layers_out[net->layer_cnt-1][neurons];
    if (temp >= temp_max) {
      temp_max = temp;
      max_c_index = neurons;
    }
  }
  return max_c_index;
}

int main(int argc, char *argv[]) {
  FILE *weight_file = fopen(argv[1], "rb");
  if (weight_file == NULL) {
    die("Could not open weight file");
  }
  FILE *spec = fopen("./network_format.txt", "r");
  if (spec == NULL) {
    die("Could not open network format file");
  }
  int layer_sizes[64];
  int layer_cnt = 0;
  char line[32];
  while (fgets(&line[0], 32, spec) != NULL) {
    char *end;
    layer_sizes[layer_cnt++] = strtoul(&line[0], &end, 10);
    if (end == &line[0]) {
      die("Invalid layer size");
    }
  }
  net_t *net = (net_t*)malloc(sizeof(net_t));
  net->layers = (layer_t**)malloc(sizeof(layer_t*) * layer_cnt);
  net->layer_cnt = layer_cnt;
  for (int i = 0; i < layer_cnt; i++) {
    net->layers[i] = (layer_t*)malloc(sizeof(layer_t));
    net->layers[i]->neuron_cnt = layer_sizes[i];
    net->layers[i]->neurons = (neuron_t**)malloc(sizeof(neuron_t*) * layer_sizes[i]);
    for (int j = 0; j < layer_sizes[i]; j++) {
      int neuron_weight_cnt = (i == 0) ? 400 : layer_sizes[i - 1];
      net->layers[i]->neurons[j] = (neuron_t*)malloc(sizeof(neuron_t));
      net->layers[i]->neurons[j]->bias = read_value(weight_file);
      net->layers[i]->neurons[j]->weight_cnt = neuron_weight_cnt;
      net->layers[i]->neurons[j]->weights = (int16_t*)malloc(sizeof(int16_t) * neuron_weight_cnt);
      for (int k = 0; k < neuron_weight_cnt; k++) {
        net->layers[i]->neurons[j]->weights[k] = read_value(weight_file);
      }
    }
  }
  //print_net(net);
  //exit(0);

  mongoc_collection_t *rot_data;
  mongoc_database_t *database;
  mongoc_cursor_t *cursor;
  mongoc_client_t *client;

  const bson_t *doc;
  bson_t *query;
  bson_t *opts;

  int test_cnt = 1080;

  mongoc_init();
  client = mongoc_client_new("mongodb://127.0.0.1/");
  database = mongoc_client_get_database(client, "capstone");
  rot_data = mongoc_database_get_collection(database, "rot_data");

  query = bson_new();
  opts = BCON_NEW("limit", BCON_INT64(test_cnt));

  cursor = mongoc_collection_find_with_opts(rot_data, query, opts, NULL);

  int pass_cnt = 0;
  while (mongoc_cursor_next(cursor, &doc)) {
    bson_iter_t iter;
    bson_iter_init(&iter, doc);
    bson_iter_find(&iter, "data");
    uint32_t image_data_len;
    const uint8_t *image_data;
    bson_iter_binary(&iter, NULL, &image_data_len, &image_data);
    bson_iter_find(&iter, "rotation");
    int32_t rotation = bson_iter_int32(&iter) / 10;
    int result = net_forward(net, image_data);
    if (rotation == result) {
      pass_cnt++;
    }
    //printf("%d\n", result);
  }

  int accuracy = (int)(((float)pass_cnt / (float)test_cnt) * 100);
  printf("Accuracy: %d%%\n", accuracy);

  bson_destroy(query);
  bson_destroy(opts);
  mongoc_collection_destroy(rot_data);
  mongoc_database_destroy(database);
  mongoc_client_destroy(client);
  mongoc_cursor_destroy(cursor);
  mongoc_cleanup();
}
