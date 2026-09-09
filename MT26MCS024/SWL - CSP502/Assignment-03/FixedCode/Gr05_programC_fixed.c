#include <stdio.h>
int main() {
  int *p;
  int x = 5;
  p = &x;
  printf("%d\n", *p);
  return 0;
}