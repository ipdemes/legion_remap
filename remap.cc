#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <mpi.h>
#include <legion.h>


#include "mapper.h"

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_SMALL_TASK_ID,
  INIT_LARGE_TASK_ID,
};

enum FieldIDs {
  FID
};

void top_level_task(const Task *,
                    const std::vector<PhysicalRegion> &,
                    Context ctx, Runtime *runtime) {

  printf("Top level task\n");

  //creating data for the small mesh

  const size_t num_elmts_small = 64;
  const size_t num_ghosts_small = 2;
  const size_t num_colors_small = 4;

  Rect<1> color_bounds_small(0, num_colors_small - 1);

  size_t max_size_small = size_t(-1) / (num_colors_small * sizeof(int));
 
  Rect<2> rect_blis_small(
    Legion::Point<2>(0, 0), Legion::Point<2>(num_colors_small - 1,
			max_size_small - 1));

  IndexSpace is_blis_small = runtime->create_index_space(ctx, rect_blis_small);

  FieldSpace fs_blis_small = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs_blis_small);
    allocator.allocate_field(sizeof(size_t), FID);
  }

  //creating data for the large mesh

  const size_t num_elmts_large = 100;
  const size_t num_ghosts_large = 4;
  const size_t num_colors_large = 9;

  Rect<1> color_bounds_large(0, num_colors_large - 1);

  size_t max_size_large = size_t(-1) / (num_colors_large * sizeof(int));

  Rect<2> rect_blis_large(
    Legion::Point<2>(0, 0), Legion::Point<2>(num_colors_large - 1,
      max_size_large - 1));

  IndexSpace is_blis_large = runtime->create_index_space(ctx, rect_blis_large);

  FieldSpace fs_blis_large = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs_blis_large);
    allocator.allocate_field(sizeof(size_t), FID);
  }

  

 

}

void init_small(const Task *,
                    const std::vector<PhysicalRegion> &,
                    Context ctx, Runtime *runtime) {
}

void init_large(const Task *,
                    const std::vector<PhysicalRegion> &,
                    Context ctx, Runtime *runtime) {
}

int main(int argc, char **argv) {

  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(INIT_SMALL_TASK_ID, "init small");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<init_small>(registrar, "init small");
  }
  {
    TaskVariantRegistrar registrar(INIT_LARGE_TASK_ID, "init large");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<init_large>(registrar, "init large");
  }


  //register custom mapper
  Runtime::add_registration_callback(mapper_registration);

  Runtime::start(argc, argv);

  printf("SUCCESS!\n");

  return 0;

}

