#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
  char name[20];
  int marks;
} Student;
void update_record(Student *s, const char *name, int marks) {
  strcpy(s->name, name);
  s->marks = marks;
}
void process_record(Student *s, const char *name, int marks) {
  update_record(s, name, marks);
  if (s->marks >= 40)
    printf("%s : PASS\n", s->name);
  else
    printf("%s : FAIL\n", s->name);
}
int main(int argc, char *argv[]) {
  Student *student = malloc(sizeof(Student));
  if (argc != 3) {
    printf("Usage: %s <name> <marks>\n", argv[0]);
    return 1;
  }
  int marks = atoi(argv[2]);
  process_record(student, argv[1], marks);
  free(student);
  return 0;
}