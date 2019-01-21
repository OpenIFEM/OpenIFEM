#include <iostream>

#include "particle.h"
#include "body.h"

int main() {
  body<particle_tl_weak> body;
  std::cout << "This is a dummy program to test if OpenIFEM "
               "is able to compile and link with mfree_iwf. "
               "Obviously it works!" << std::endl;
  return EXIT_SUCCESS;
}
