#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <legion.h>
#include <mpi.h>

#include "mapper.h"

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_SMALL_TASK_ID,
  INIT_LARGE_TASK_ID,
  FILL_PART_TASK_ID,
  REMAP_TASK_ID,
};

enum FieldIDs { FID, PART_FID1, PART_FID2, PART_FID3, PART_FID4 };

void top_level_task(const Task *, const std::vector<PhysicalRegion> &,
                    Context ctx, Runtime *runtime) {

  printf("Top level task\n");
  //------------------------------------------------------------------------
  // creating data for the small mesh
  //------------------------------------------------------------------------

  const size_t num_elmts_small = 64;
  const size_t num_ghosts_small = 2;
  const size_t num_colors_small = 4;

  Rect<1> color_bounds_small(0, num_colors_small - 1);

  IndexSpaceT<1> color_is_small =
      runtime->create_index_space(ctx, color_bounds_small);
  size_t max_size_small = size_t(-1) / (num_colors_small * sizeof(int));

  Rect<2> rect_blis_small(
      Legion::Point<2>(0, 0),
      Legion::Point<2>(num_colors_small - 1, max_size_small - 1));

  IndexSpace is_blis_small = runtime->create_index_space(ctx, rect_blis_small);

  FieldSpace fs_blis_small = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
        runtime->create_field_allocator(ctx, fs_blis_small);
    allocator.allocate_field(sizeof(size_t), FID);
  }

  LogicalRegion small_lr =
      runtime->create_logical_region(ctx, is_blis_small, fs_blis_small);

  Legion::Transform<2, 1> ret;
  ret.rows[0].x = 1;
  ret.rows[1].x = 0;

  Rect<2> extend_small(
      Legion::Point<2>(0, 0),
      Legion::Point<2>(0, num_elmts_small + num_ghosts_small - 1));

  IndexPartition small_ip = runtime->create_partition_by_restriction(
      ctx, is_blis_small, color_is_small, ret, extend_small,
      DISJOINT_COMPLETE_KIND);

  LogicalPartition small_lp =
      runtime->get_logical_partition(small_lr, small_ip);

  ArgumentMap idx_arg_map;
  IndexLauncher init_small_launcher(INIT_SMALL_TASK_ID, color_is_small,
                                    TaskArgument(NULL, 0), idx_arg_map);
  init_small_launcher.add_region_requirement(
      RegionRequirement(small_lp, 0, WRITE_DISCARD, EXCLUSIVE, small_lr));
  init_small_launcher.region_requirements[0].add_field(FID);
  runtime->execute_index_space(ctx, init_small_launcher);

  //------------------------------------------------------------------------
  // creating data for the large mesh
  //------------------------------------------------------------------------

  const size_t num_elmts_large = 100;
  const size_t num_ghosts_large = 4;
  const size_t num_colors_large = 9;

  Rect<1> color_bounds_large(0, num_colors_large - 1);
  IndexSpaceT<1> color_is_large =
      runtime->create_index_space(ctx, color_bounds_large);

  size_t max_size_large = size_t(-1) / (num_colors_large * sizeof(int));

  Rect<2> rect_blis_large(
      Legion::Point<2>(0, 0),
      Legion::Point<2>(num_colors_large - 1, max_size_large - 1));

  IndexSpace is_blis_large = runtime->create_index_space(ctx, rect_blis_large);

  FieldSpace fs_blis_large = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
        runtime->create_field_allocator(ctx, fs_blis_large);
    allocator.allocate_field(sizeof(size_t), FID);
    allocator.allocate_field(sizeof(Rect<2>), PART_FID1);
    allocator.allocate_field(sizeof(Rect<2>), PART_FID2);
    allocator.allocate_field(sizeof(Rect<2>), PART_FID3);
    allocator.allocate_field(sizeof(Rect<2>), PART_FID4);
  }

  LogicalRegion large_lr =
      runtime->create_logical_region(ctx, is_blis_large, fs_blis_large);

  Rect<2> extend_large(
      Legion::Point<2>(0, 0),
      Legion::Point<2>(0, num_elmts_large + num_ghosts_large - 1));

  IndexPartition large_ip = runtime->create_partition_by_restriction(
      ctx, is_blis_large, color_is_large, ret, extend_large,
      DISJOINT_COMPLETE_KIND);

  LogicalPartition large_lp =
      runtime->get_logical_partition(large_lr, large_ip);

  IndexLauncher init_large_launcher(INIT_LARGE_TASK_ID, color_is_large,
                                    TaskArgument(NULL, 0), idx_arg_map);
  init_large_launcher.add_region_requirement(
      RegionRequirement(large_lp, 0, WRITE_DISCARD, EXCLUSIVE, large_lr));
  init_large_launcher.region_requirements[0].add_field(FID);
  runtime->execute_index_space(ctx, init_large_launcher);

  //------------------------------------------------------------------------
  // create overlaping partition for the small mesh
  //------------------------------------------------------------------------

  IndexPartition overlap_ip;

  {
    Rect<2> extend2(Legion::Point<2>(0, 0), Legion::Point<2>(0, 0));

    IndexPartition color_ip = runtime->create_partition_by_restriction(
        ctx, is_blis_large, color_is_large, ret, extend2,
        DISJOINT_COMPLETE_KIND);

    LogicalPartition color_lp =
        runtime->get_logical_partition(large_lr, color_ip);

    // Launch the task that fills PART_FID
    IndexLauncher fill_part_launcher(FILL_PART_TASK_ID, color_is_large,
                                     TaskArgument(NULL, 0), idx_arg_map);
    fill_part_launcher.add_region_requirement(
        RegionRequirement(color_lp, 0, WRITE_DISCARD, EXCLUSIVE, large_lr));
    fill_part_launcher.region_requirements[0].add_field(PART_FID1);
    fill_part_launcher.region_requirements[0].add_field(PART_FID2);
    fill_part_launcher.region_requirements[0].add_field(PART_FID3);
    fill_part_launcher.region_requirements[0].add_field(PART_FID4);
    runtime->execute_index_space(ctx, fill_part_launcher);
    IndexPartition ip1 = runtime->create_partition_by_image_range(
        ctx, is_blis_small, color_lp,
        runtime->get_parent_logical_region(color_lp), PART_FID1, color_is_large,
        ALIASED_INCOMPLETE_KIND);

    IndexPartition ip2 = runtime->create_partition_by_image_range(
        ctx, is_blis_small, color_lp,
        runtime->get_parent_logical_region(color_lp), PART_FID2, color_is_large,
        ALIASED_INCOMPLETE_KIND);

    IndexPartition ip3 = runtime->create_partition_by_image_range(
        ctx, is_blis_small, color_lp,
        runtime->get_parent_logical_region(color_lp), PART_FID3, color_is_large,
        ALIASED_INCOMPLETE_KIND);

    IndexPartition ip4 = runtime->create_partition_by_image_range(
        ctx, is_blis_small, color_lp,
        runtime->get_parent_logical_region(color_lp), PART_FID4, color_is_large,
        ALIASED_INCOMPLETE_KIND);

    IndexPartition iu1 = runtime->create_partition_by_union(
        ctx, is_blis_small, ip1, ip2, color_is_large, ALIASED_KIND);

    IndexPartition iu2 = runtime->create_partition_by_union(
        ctx, is_blis_small, ip3, ip4, color_is_large, ALIASED_KIND);

    overlap_ip = runtime->create_partition_by_union(
        ctx, is_blis_small, iu1, iu2, color_is_large, ALIASED_COMPLETE_KIND);
  }

  LogicalPartition overlap_lp =
      runtime->get_logical_partition(small_lr, overlap_ip);

  //------------------------------------------------------------------------
  // launch remap task
  //------------------------------------------------------------------------
  IndexLauncher remap_launcher(REMAP_TASK_ID, color_is_large,
                               TaskArgument(NULL, 0), idx_arg_map);
  remap_launcher.add_region_requirement(
      RegionRequirement(large_lp, 0, READ_WRITE, EXCLUSIVE, large_lr));
  remap_launcher.region_requirements[0].add_field(FID);

  remap_launcher.add_region_requirement(
      RegionRequirement(overlap_lp, 0, READ_ONLY, EXCLUSIVE, small_lr));
  remap_launcher.region_requirements[1].add_field(FID);

  runtime->execute_index_space(ctx, remap_launcher);
}//top level task

//------------------------------------------------------------------------
void init_small_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions, Context ctx,
                     Runtime *runtime) {

  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  auto color = task->index_point.point_data[0];
  const FieldAccessor<WRITE_DISCARD, size_t, 2> acc(regions[0], FID);
  auto rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  for (PointInRectIterator<2> pir(rect); pir(); pir++) {
    acc[*pir] = 4 * color;
  }

  std::cout << "IRNA DEBUG rect = " << rect << std::endl;
} // init_small

//------------------------------------------------------------------------
void init_large_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions, Context ctx,
                     Runtime *runtime) {

  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  auto color = task->index_point.point_data[0];
  const FieldAccessor<WRITE_DISCARD, size_t, 2> acc(regions[0], FID);
  auto rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  for (PointInRectIterator<2> pir(rect); pir(); pir++) {
    acc[*pir] = 9 * color;
  }

} // init large

//------------------------------------------------------------------------
void fill_part_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions, Context ctx,
                    Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 4);

  std::cout << "Inside of the Fill_part_task" << std::endl;
  auto color = task->index_point.point_data[0];

  const FieldAccessor<WRITE_DISCARD, Rect<2>, 2> acc1(regions[0], PART_FID1);
  const FieldAccessor<WRITE_DISCARD, Rect<2>, 2> acc2(regions[0], PART_FID2);
  const FieldAccessor<WRITE_DISCARD, Rect<2>, 2> acc3(regions[0], PART_FID3);
  const FieldAccessor<WRITE_DISCARD, Rect<2>, 2> acc4(regions[0], PART_FID4);

  auto rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());

  PointInRectIterator<2> pir(rect);

  switch (color) {
  case 0: {
    acc1[*pir] = {{0, 0}, {0, 65}};
    break;
  }
  case 1: {
    acc1[*pir] = {{0, 0}, {0, 65}};
    acc2[*pir] = {{1, 0}, {1, 65}};
    break;
  }
  case 2: {
    acc2[*pir] = {{1, 0}, {1, 65}};
    break;
  }
  case 3: {
    acc1[*pir] = {{0, 0}, {0, 65}};
    acc3[*pir] = {{2, 0}, {2, 45}};
    break;
  }
  case 4: {
    acc1[*pir] = {{0, 0}, {0, 65}};
    acc2[*pir] = {{1, 0}, {1, 65}};
    acc3[*pir] = {{2, 0}, {2, 45}};
    acc4[*pir] = {{3, 0}, {3, 15}};
    break;
  }
  case 5: {
    acc2[*pir] = {{1, 0}, {1, 65}};
    acc4[*pir] = {{3, 0}, {3, 15}};
    break;
  }
  case 6: {
    acc3[*pir] = {{2, 0}, {2, 45}};
    break;
  }
  case 7: {
    acc3[*pir] = {{2, 0}, {2, 45}};
    acc4[*pir] = {{3, 0}, {3, 15}};
    break;
  }
  case 8: {
    acc4[*pir] = {{3, 0}, {3, 15}};
    break;
  }
  }

} // fill_part_task

//------------------------------------------------------------------------
void remap_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {

  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);

  const FieldAccessor<READ_WRITE, size_t, 2> acc_l(regions[0], FID);

  auto rect_l = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());

  const FieldAccessor<READ_ONLY, size_t, 2> acc_s(regions[1], FID);

  auto rect_s = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

   auto color = task->index_point.point_data[0];

  for (PieceIterator pir(regions[1], FID, true); pir(); pir++) {
    std::cout << "IRINA DEBUG color = " <<color<< "overlap rect = " << *pir << std::endl;
    for (PointInRectIterator<2> pir2(*pir); pir2(); pir2++) {
  //    std::cout << acc_s[*pir2] << std::endl;
    }
  }

} // remap task

//------------------------------------------------------------------------
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
    Runtime::preregister_task_variant<init_small_task>(registrar, "init small");
  }
  {
    TaskVariantRegistrar registrar(INIT_LARGE_TASK_ID, "init large");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<init_large_task>(registrar, "init large");
  }
  {
    TaskVariantRegistrar registrar(FILL_PART_TASK_ID, "fill partition");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<fill_part_task>(registrar,
                                                      "fill_partition");
  }
  {
    TaskVariantRegistrar registrar(REMAP_TASK_ID, "remap");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<remap_task>(registrar, "remap");
  }

  // register custom mapper
  Runtime::add_registration_callback(mapper_registration);

  Runtime::start(argc, argv);

  printf("SUCCESS!\n");

  return 0;
}
