#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

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

static int16_t read_value(FILE *weight_file) {
  uint8_t bytes[2];
  if (fread(&bytes[0], 1, 2, weight_file) != 2) {
    die("Could not read value");
  }
  return (((int16_t)bytes[0]) << 8) | ((int16_t)bytes[1]);
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
  int layer_cnt;
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
      int neuron_weight_cnt = (j == 0) ? 400 : layer_sizes[i - 1];
      net->layers[i]->neurons[j] = (neuron_t*)malloc(sizeof(neuron_t));
      net->layers[i]->neurons[j]->bias = read_value(weight_file);
      net->layers[i]->neurons[j]->weight_cnt = neuron_weight_cnt;
      net->layers[i]->neurons[j]->weights = (int16_t*)malloc(sizeof(int16_t) * neuron_weight_cnt);
      for (int k = 0; k < neuron_weight_cnt; k++) {
        net->layers[i]->neurons[j]->weights[k] = read_value(weight_file);
        printf("%d\n", net->layers[i]->neurons[j]->weights[k])
      }
    }
  }
}
