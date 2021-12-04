#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include <mongoc/mongoc.h>
#include <bson/bson.h>

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
  return value / 4096;
}

static int16_t read_value(FILE *weight_file) {
  uint8_t bytes[2];
  if (fread(&bytes[0], 1, 2, weight_file) != 2) {
    die("Could not read value");
  }
  return (((int16_t)bytes[0]) << 8) | ((int16_t)bytes[1]);
}

static int net_forward(net_t *net, uint8_t *sub_image) {
  double *layers_out[net->layer_cnt];
  for (int i = 0; i < net->layer_cnt; i++) {
    layers_out[i] = (double*)malloc(sizeof(double) * net->layers[i]->neuron_cnt);
    for (int j = 0; j < net->layers[i]->neuron_cnt; j++) {
      double activation = to_double(net->layers[i]->neurons[j]->bias);
      for (int k = 0; k < net->layers[i]->neurons[j]->weight_cnt; k++) {
        double weight = to_double(net->layers[i]->neurons[j]->weights[k]);
        if (i == 0) {
          activation += weight * ((double)sub_image[k] / (double)1);
        } else {
          activation += weight * layers_out[i - 1][k];
        }
      }
      if (i != net->layer_cnt - 1) {
        layers_out[i][j] = tanh(activation);
      } else {
        layers_out[i][j] = activation;
      }
    }
  }
  double e_sum = 0;
  int output_size = net->layers[net->layer_cnt - 1]->neuron_cnt;
  for (int i = 0; i < output_size; i++) {
    e_sum += exp(layers_out[net->layer_cnt - 1][i]);
  }
  int max_index = 0;
  double current_max = -DBL_MAX;
  for (int i = 0; i < output_size; i++) {
    double prob = layers_out[net->layer_cnt - 1][i] / e_sum;
    if (prob > current_max) {
      current_max = prob;
      max_index = i;
    }
  }
  return i;
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
      int neuron_weight_cnt = (j == 0) ? 400 : layer_sizes[i - 1];
      net->layers[i]->neurons[j] = (neuron_t*)malloc(sizeof(neuron_t));
      net->layers[i]->neurons[j]->bias = read_value(weight_file);
      net->layers[i]->neurons[j]->weight_cnt = neuron_weight_cnt;
      net->layers[i]->neurons[j]->weights = (int16_t*)malloc(sizeof(int16_t) * neuron_weight_cnt);
      for (int k = 0; k < neuron_weight_cnt; k++) {
        net->layers[i]->neurons[j]->weights[k] = read_value(weight_file);
      }
    }
  }

  mongoc_collection_t *rot_data;
  mongoc_database_t *database;
  mongoc_cursor_t *cursor;
  mongoc_client_t *client;

  const bson_t *doc;
  bson_t *query;
  bson_t *opts;

  mongoc_init();
  client = mongoc_client_new("mongodb://127.0.0.1/");
  database = mongoc_client_get_database(client, "capstone");
  rot_data = mongoc_database_get_collection(database, "rot_data");

  query = bson_new();
  opts = BCON_NEW("limit", BCON_INT64(1080));

  cursor = mongoc_collection_find_with_opts(rot_data, query, opts, NULL);

  while (mongoc_cursor_next(cursor, &doc)) {
    bson_iter_t iter;
    if (bson_iter_init(&iter, doc) && bson_iter_find(&iter, "data")) {
      uint32_t image_data_len;
      const uint8_t *image_data;
      bson_iter_binary(&iter, NULL, &image_data_len, &image_data);
      printf("Found image with buffer length %d\n", image_data_len);
    }
  }

  bson_destroy(query);
  bson_destroy(opts);
  mongoc_collection_destroy(rot_data);
  mongoc_database_destroy(database);
  mongoc_client_destroy(client);
  mongoc_cursor_destroy(cursor);
  mongoc_cleanup();
}
