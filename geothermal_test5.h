#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <vector>
#include <cmath>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/symmetric_tensor.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/matrix_out.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include "interpolation.h"

namespace LA {
using namespace dealii::LinearAlgebraPETSc;
#define USE_PETSC_LA

}  // namespace LA

using namespace dealii;

template <int dim>
class CoupledTH {
 public:
  CoupledTH(const unsigned int degree);
  ~CoupledTH();
  void run();

 private:
  void make_grid_and_dofs();
  void setup_P_system();
  void setup_T_system();
  void assemble_P_system();
  void assemble_T_system();
  void linear_solve_P();
  void linear_solve_T();
  void output_results(LA::MPI::Vector&, std::string) const;
  void output_results2(LA::MPI::Vector&, std::string) const;

  MPI_Comm mpi_communicator; //in mpi.h
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  ConditionalOStream pcout; // in parallel program, each of MPI processes will write to the screen. This class solves this.
  TimerOutput computing_timer; //generate formatted output

  parallel::distributed::Triangulation<dim> triangulation;  // grid or meshes
  Triangulation<dim-2,dim> triangulation_line;
  
  DoFHandler<dim> dof_handler;  // degree of freedom on vertices, lines, etc
  FE_Q<dim> fe;                 // Lagrange interpolation polynomials
  QGauss<dim> quadrature_formula; //quadrature rules for numerical integration
  QGauss<dim - 1> face_quadrature_formula;
  QGauss<dim - 2> line_quadrature_formula;

  QGauss<dim> quadrature_formula_T;
  QGauss<dim - 1> face_quadrature_formula_T;

  DoFHandler<dim> dof_handler_T;
  dealii::FESystem<dim> fe_T;

  DoFHandler<dim-2,dim> dof_handler_line;
  FE_Q<dim-2,dim> fe_line;

  IndexSet locally_owned_dofs; // subset of indices
  IndexSet locally_relevant_dofs;
  IndexSet locally_owned_dofs_T; // subset of indices
  IndexSet locally_relevant_dofs_T;

  const unsigned int degree;  // element degree

  LA::MPI::SparseMatrix P_system_matrix;  // M_P + K_P
  LA::MPI::SparseMatrix T_system_matrix;  // M_T + K_T
  LA::MPI::Vector P_system_rhs;           // right hand side of P
                                          // system
  LA::MPI::Vector T_system_rhs;           // right hand side of T
                                          // system

  LA::MPI::Vector P_locally_relevant_solution;  // P solution at n
  LA::MPI::Vector T_locally_relevant_solution;  // T solution at n

  LA::MPI::Vector old_P_locally_relevant_solution;  // P solution at n -1
  LA::MPI::Vector old_T_locally_relevant_solution;  // T solution at n -1

  Vector<double> initial_P_solution;  // P solution at 0
  Vector<double> initial_T_solution;  // T solution at 0

  std::map<types::global_dof_index, types::global_dof_index> dof_dof_map_Ttu;
  std::map<types::global_dof_index, types::global_dof_index> dof_dof_map_Ts;
  std::map<types::global_dof_index, types::global_dof_index> dof_dof_map_line;
  std::map<types::global_dof_index, types::global_dof_index> samez_dof_dof_map;
  std::map<types::global_dof_index,std::vector<double>> dof_vert_map;
  std::vector<types::global_dof_index> dof_appeared_Ttu;
  std::map<types::global_dof_index, double> dummy_map;

  double time;                   // t
  unsigned int timestep_number;  // n_t
  std::vector<double> time_sequence;

  double total_time;
  int n_time_step;
  double time_step;
  double period;

  unsigned int n_q_points_P;

  unsigned int P_iteration_namber;
  unsigned int T_iteration_namber;
  unsigned int n_P_max_iteration = EquationData::n_g_P_max_iteration;
  unsigned int n_T_max_iteration = EquationData::n_g_T_max_iteration;
  double P_tol_residual = EquationData::g_P_tol_residual;
  double T_tol_residual = EquationData::g_T_tol_residual;
  Interpolation<3> data_interpolation;
};

template <int dim>
CoupledTH<dim>::CoupledTH(const unsigned int degree)  // initialization
    : mpi_communicator(MPI_COMM_WORLD),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      triangulation(mpi_communicator),
      dof_handler(triangulation),
      dof_handler_T(triangulation),
      dof_handler_line(triangulation_line),
      fe(degree),
      fe_T(FE_Q<dim>(degree),2),
      fe_line(degree),
      degree(degree),
      quadrature_formula(degree + 1),
      face_quadrature_formula(degree + 1),
      line_quadrature_formula(degree + 1),
      quadrature_formula_T(degree + 1),
      face_quadrature_formula_T(degree + 1),
      time(0.0),
      timestep_number(0),
      period(EquationData::g_period),
      P_iteration_namber(0),
      T_iteration_namber(0),
      pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
      computing_timer(mpi_communicator, pcout, TimerOutput::summary,
                      TimerOutput::wall_times), // Output wall clock times.
      data_interpolation(EquationData::dimension_x, EquationData::dimension_y,
                         EquationData::dimension_z,
                         EquationData::file_name_interpolation) {
  if (EquationData::is_linspace) {
    total_time = EquationData::g_total_time;
    n_time_step = EquationData::g_n_time_step;
    time_sequence = linspace(0.0, total_time, n_time_step);
    time_step = time_sequence[1] - time_sequence[0];
  } else {
    time_sequence = EquationData::g_time_sequence;
    n_time_step = time_sequence.size();
    total_time = time_sequence[n_time_step - 1];
    time_step = time_sequence[1] - time_sequence[0];
  }
}

template <int dim>
CoupledTH<dim>::~CoupledTH() {
  dof_handler.clear();
}

template <int dim>
void CoupledTH<dim>::make_grid_and_dofs() {
  std::map<std::vector<double>, types::global_dof_index> vert_dof_map;
  std::vector<double> exist_z;
  std::vector<types::global_dof_index> exist_dof;
  double min_z=-120;

  GridIn<dim> gridin;  // instantiate a gridinput
  gridin.attach_triangulation(triangulation);
  std::ifstream f(EquationData::mesh_file_name);
  gridin.read_msh(f);
  dof_handler.distribute_dofs(fe);  // distribute dofs to grid globle
  dof_handler_T.distribute_dofs(fe_T);
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_owned_dofs_T = dof_handler_T.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  DoFTools::extract_locally_relevant_dofs(dof_handler_T, locally_relevant_dofs_T);

  pcout << "Number of active cells: " << triangulation.n_active_cells() // the total number of active cells
        << " (on " << triangulation.n_levels() << " levels)" << std::endl // returns the level of the most refined active cell plus one (all cells with coarse, unrefined mesh)
        << "Number of degrees of freedom of P: " << dof_handler.n_dofs()
        << " Number of degrees of freedom of T: " << dof_handler_T.n_dofs()
        << std::endl
        << std::endl; // n_dofs = number of shape functions that span this space

  initial_P_solution.reinit(dof_handler.n_dofs());
  initial_T_solution.reinit(dof_handler_T.n_dofs());
  old_P_locally_relevant_solution.reinit(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  old_T_locally_relevant_solution.reinit(
      locally_owned_dofs_T, locally_relevant_dofs_T, mpi_communicator);

  GridIn<dim-2,dim> gridin_line;  // instantiate a gridinput
  gridin_line.attach_triangulation(triangulation_line);
  std::ifstream f_line(EquationData::mesh_file_name_line);
  gridin_line.read_msh(f_line);
  dof_handler_line.distribute_dofs(fe_line); 
  typename DoFHandler<dim-2,dim>::active_cell_iterator cell = dof_handler_line.begin_active(),
  endc = dof_handler_line.end();
  std::vector<types::global_dof_index> dof_appeared;
  std::vector<types::global_dof_index> start_point;
  std::vector<types::global_dof_index> first_point;
  std::vector<types::global_dof_index> second_point;
  FEValues<dim-2,dim> fe_values_line(fe_line, line_quadrature_formula, update_values);
  std::vector<types::global_dof_index> local_dof_indices_line(fe_values_line.dofs_per_cell);
  for (; cell != endc; ++cell) {
    cell->get_dof_indices(local_dof_indices_line);
    fe_values_line.reinit(cell);
    int i = 0;
    for (const unsigned int j : fe_values_line.dof_indices()) {
      Point<3> &v = cell->vertex(fe_line.system_to_component_index(j).second);
      std::vector<double> vert_coords_vec; 
      vert_coords_vec.push_back(v(0)); 
      vert_coords_vec.push_back(v(1)); 
      vert_coords_vec.push_back(v(2));
      if (i==0) {
        first_point.push_back(local_dof_indices_line[j]);
      } else {
        second_point.push_back(local_dof_indices_line[j]);
      }
      if(std::find(std::begin(EquationData::start_point), std::end(EquationData::start_point), vert_coords_vec) != std::end(EquationData::start_point) &&
        std::find(start_point.begin(), start_point.end(), local_dof_indices_line[j]) == start_point.end()){
        start_point.push_back(local_dof_indices_line[j]);
        // pcout << "Start point:" << local_dof_indices_line[j] << std::endl;
      }
      if(std::find(dof_appeared.begin(), dof_appeared.end(), local_dof_indices_line[j]) == dof_appeared.end()) {
        dof_appeared.push_back(local_dof_indices_line[j]); 
        // pcout << "Dof:" << local_dof_indices_line[j] << ", coords:" << v(0) << ", " << v(1) << ", "<< v(2) << std::endl;
        std::pair<std::vector<double>, types::global_dof_index> vd = std::make_pair(vert_coords_vec, local_dof_indices_line[j]);
        vert_dof_map.emplace(vd);
        if (v(2)>min_z) {
          if(std::find(exist_z.begin(), exist_z.end(), v(2)) == exist_z.end()) {
            exist_z.push_back(v(2));
            exist_dof.push_back(local_dof_indices_line[j]);
          } else {
            auto it = std::find(exist_z.begin(), exist_z.end(), v(2));
            int index = it - exist_z.begin();
            auto temp_dof = exist_dof.at(index);
            std::pair<types::global_dof_index,types::global_dof_index> temp_vd0 = std::make_pair(temp_dof, local_dof_indices_line[j]);
            std::pair<types::global_dof_index,types::global_dof_index> temp_vd1 = std::make_pair(local_dof_indices_line[j], temp_dof);
            samez_dof_dof_map.emplace(temp_vd0);
            samez_dof_dof_map.emplace(temp_vd1);
            exist_z.erase(exist_z.begin()+index);
            exist_dof.erase(exist_dof.begin()+index);
          }
        }
      }
      i++;
    }
  }

  typename DoFHandler<dim>::active_cell_iterator cell_T = dof_handler_T.begin_active(),
  endc_T = dof_handler_T.end();
  FEValues<dim> fe_values(fe_T, quadrature_formula_T, update_values);
  std::vector<types::global_dof_index> dof_appeared_Ts;
  std::vector<types::global_dof_index> dof_appeared_dummy;
  std::vector<types::global_dof_index> local_dof_indices_T(fe_values.dofs_per_cell);
  for (; cell_T != endc_T; ++cell_T) {
    if (cell_T->is_locally_owned()) {
      cell_T->get_dof_indices(local_dof_indices_T);
      fe_values.reinit(cell_T);
      for (const unsigned int j : fe_values.dof_indices()) {
        const unsigned int component_j = fe_T.system_to_component_index(j).first;
        Point<3> &v_T = cell_T->vertex(fe_T.system_to_component_index(j).second);
        std::vector<double> vert_coords_vec; 
        vert_coords_vec.push_back(v_T(0)); 
        vert_coords_vec.push_back(v_T(1)); 
        vert_coords_vec.push_back(v_T(2));

        if(vert_dof_map.find(vert_coords_vec) != vert_dof_map.end() && std::find(dof_appeared_Ts.begin(), dof_appeared_Ts.end(), local_dof_indices_T[j]) == dof_appeared_Ts.end() && component_j==0) {
          dof_appeared_Ts.push_back(local_dof_indices_T[j]);
          types::global_dof_index found_dof = vert_dof_map.at(vert_coords_vec);
          std::pair<types::global_dof_index, types::global_dof_index> vd2 = std::make_pair(found_dof,local_dof_indices_T[j]);
          dof_dof_map_Ts.emplace(vd2);
          // pcout << "Dof1_Ts:" << found_dof << ", Dof2_Ts:" << local_dof_indices_T[j] << std::endl;
        }

        if(vert_dof_map.find(vert_coords_vec) != vert_dof_map.end() && std::find(dof_appeared_Ttu.begin(), dof_appeared_Ttu.end(), local_dof_indices_T[j]) == dof_appeared_Ttu.end() && component_j==1) {
          dof_appeared_Ttu.push_back(local_dof_indices_T[j]);
          types::global_dof_index found_dof = vert_dof_map.at(vert_coords_vec);
          std::pair<types::global_dof_index, types::global_dof_index> vd2 = std::make_pair(found_dof,local_dof_indices_T[j]);
          dof_dof_map_Ttu.emplace(vd2);
          // pcout << "Dof1_Ttu:" << found_dof << ", Dof2_Ttu:" << local_dof_indices_T[j] << std::endl;
          if(std::find(start_point.begin(), start_point.end(), found_dof) != start_point.end()) {
            std::replace (start_point.begin(), start_point.end(), found_dof, local_dof_indices_T[j]);
          }
          if(std::find(first_point.begin(), first_point.end(), found_dof) != first_point.end()) {
            std::replace (first_point.begin(), first_point.end(), found_dof, local_dof_indices_T[j]);
          }
          if(std::find(second_point.begin(), second_point.end(), found_dof) != second_point.end()) {
            std::replace (second_point.begin(), second_point.end(), found_dof, local_dof_indices_T[j]);
          }
          std::pair<types::global_dof_index,std::vector<double>> vd = std::make_pair(local_dof_indices_T[j],vert_coords_vec);
          dof_vert_map.emplace(vd);
        }

        if(vert_dof_map.find(vert_coords_vec) == vert_dof_map.end() && std::find(dof_appeared_dummy.begin(), dof_appeared_dummy.end(), local_dof_indices_T[j]) == dof_appeared_dummy.end() && component_j==1) {
          dof_appeared_dummy.push_back(local_dof_indices_T[j]);
          std::pair<types::global_dof_index, double> vd2 = std::make_pair(local_dof_indices_T[j], 0);
          dummy_map.emplace(vd2);
        }
      }
    }
  }

  for(auto start_point_i : start_point) {
    types::global_dof_index temp_point = start_point_i;
    auto it1 = std::find(first_point.begin(), first_point.end(), temp_point);
    auto it2 = std::find(second_point.begin(), second_point.end(), temp_point);
    bool is_first = (it1 != first_point.end());
    bool is_second = (it2 != second_point.end());
    // pcout<<temp_point<<std::endl;
    while(is_first || is_second) {
      if(is_first) {
        int index = it1 - first_point.begin();
        std::pair<types::global_dof_index, types::global_dof_index> vd = std::make_pair(temp_point, second_point.at(index));
        dof_dof_map_line.emplace(vd);
        temp_point = second_point.at(index);
        first_point.erase(it1);
        second_point.erase(second_point.begin()+index);
      } else {
        int index = it2 - second_point.begin();
        std::pair<types::global_dof_index, types::global_dof_index> vd = std::make_pair(temp_point, first_point.at(index));
        dof_dof_map_line.emplace(vd);
        temp_point = first_point.at(index);
        first_point.erase(first_point.begin()+index);
        second_point.erase(it2);
      }
      it1 = std::find(first_point.begin(), first_point.end(), temp_point);
      it2 = std::find(second_point.begin(), second_point.end(), temp_point);
      is_first = (it1 != first_point.end());
      is_second = (it2 != second_point.end());
      // pcout<<temp_point<<std::endl;
    }
  }

  for(auto start_point_i : start_point) {
    std::pair<types::global_dof_index, double> vd2 = std::make_pair(start_point_i, 273.15 + 33);
    dummy_map.emplace(vd2);
  }
  //for(auto start_point_i : dof_appeared_Ttu) {
  //  std::pair<types::global_dof_index, double> vd2 = std::make_pair(start_point_i, 273.15 + 33);
  //  dummy_map.emplace(vd2);
  //}
}

template <int dim>
void CoupledTH<dim>::setup_P_system() {
  P_locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs,
                                     mpi_communicator);

  P_system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityTools::distribute_sparsity_pattern(
      dsp, locally_owned_dofs, mpi_communicator, locally_relevant_dofs);

  // forming system matrixes and initialize these matrixesy
  P_system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                         mpi_communicator);
}

template <int dim>
void CoupledTH<dim>::setup_T_system() {
  T_locally_relevant_solution.reinit(locally_owned_dofs_T, locally_relevant_dofs_T,
                                     mpi_communicator);

  T_system_rhs.reinit(locally_owned_dofs_T, mpi_communicator);
  DynamicSparsityPattern dsp(locally_relevant_dofs_T);
  DoFTools::make_sparsity_pattern(dof_handler_T, dsp);
  SparsityTools::distribute_sparsity_pattern(
      dsp, locally_owned_dofs_T, mpi_communicator, locally_relevant_dofs_T);

  // forming system matrixes and initialize these matrixesy
  T_system_matrix.reinit(locally_owned_dofs_T, locally_owned_dofs_T, dsp,
                         mpi_communicator);
}

template <int dim>
void CoupledTH<dim>::assemble_P_system() {
  cbgeo::Clock timer;
  timer.tick();
  P_system_rhs=0;
  P_system_matrix=0;
  P_locally_relevant_solution=0;

  // Getting fe values
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                   update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

  // define loop number
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  n_q_points_P = n_q_points;

  // store the value at previous step at q_point for P
  std::vector<double> old_P_sol_values(n_q_points);
  std::vector<Tensor<1, dim>> old_P_sol_grads(n_q_points);

  // store the rhs and bd and old solution value at q_point of element for P
  std::vector<double> P_source_values(n_q_points);
  std::vector<double> QP_bd_values(n_face_q_points);

  //  local element matrix
  FullMatrix<double> P_cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> P_cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> P_local_dof_indices(dofs_per_cell);

  // boudnary condition and source term
  EquationData::PressureSourceTerm<dim> P_source_term;
  EquationData::PressureNeumanBoundaryValues<dim> QP_boundary;
  EquationData::PressureDirichletBoundaryValues<dim> P_boundary;

  // loop for cell
  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  for (; cell != endc; ++cell) {
    if (cell->is_locally_owned()) {  // only assemble the system on cells that
                                     // acturally
                                     // belong to this MPI process

      // initialization
      P_cell_matrix = 0;
      P_cell_rhs = 0;
      fe_values.reinit(cell);

      // get the values at gauss point old solution from the system
      if (time < 1e-8) {
        fe_values.get_function_values(initial_P_solution, old_P_sol_values);
        fe_values.get_function_gradients(initial_P_solution, old_P_sol_grads);
      } else {
        fe_values.get_function_values(old_P_locally_relevant_solution,
                                      old_P_sol_values);
        fe_values.get_function_gradients(old_P_locally_relevant_solution,
                                         old_P_sol_grads);
      }

      // get source term value at the gauss point
      P_source_term.set_time(time);
      P_source_term.value_list(fe_values.get_quadrature_points(),
                               P_source_values);  // 一列q个

      // loop for q_point ASSMBLING CELL METRIX (weak form equation writing)
      for (unsigned int q = 0; q < n_q_points; ++q) {

        const auto P_quadrature_coord = fe_values.quadrature_point(q);

        // 3d interp
        EquationData::g_perm = data_interpolation.value(P_quadrature_coord[0],
                                                        P_quadrature_coord[1],
                                                        P_quadrature_coord[2]);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const Tensor<1, dim> grad_phi_i_P = fe_values.shape_grad(i, q);
          const double phi_i_P = fe_values.shape_value(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const Tensor<1, dim> grad_phi_j_P = fe_values.shape_grad(j, q);
            const double phi_j_P = fe_values.shape_value(j, q);
            // mass matrix
            P_cell_matrix(i, j) += (phi_i_P * phi_j_P * fe_values.JxW(q));
            // stiff matrix
            P_cell_matrix(i, j) +=
                (time_step * EquationData::g_perm * EquationData::g_B_w *
                 grad_phi_i_P * grad_phi_j_P * fe_values.JxW(q));
          }
          P_cell_rhs(i) += (time_step * phi_i_P * P_source_values[q] +
                            time_step * grad_phi_i_P * (Point<dim>(0, 0, 1)) *
                                (-EquationData::g_B_w * EquationData::g_perm *
                                 EquationData::g_P_grad) +
                            phi_i_P * old_P_sol_values[q]) *
                           fe_values.JxW(q);
        }
      }

      for (unsigned int face_no = 0;
           face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {
        if (cell->at_boundary(face_no)) {
          for (int bd_i = 0; bd_i < EquationData::g_num_QP_bnd_id; ++bd_i) {

            if (cell->face(face_no)->boundary_id() ==
                EquationData::g_QP_bnd_id[bd_i]) {
              fe_face_values.reinit(cell, face_no);

              // get boundary condition
              QP_boundary.get_bd_i(bd_i);
              QP_boundary.set_time(time);
              QP_boundary.set_boundary_id(*(EquationData::g_QP_bnd_id + bd_i));
              QP_boundary.value_list(fe_face_values.get_quadrature_points(),
                                     QP_bd_values);

              for (unsigned int q = 0; q < n_face_q_points; ++q) {

                const auto P_face_quadrature_coord =
                    fe_face_values.quadrature_point(q);

                // 3d interp
                EquationData::g_perm = data_interpolation.value(
                    P_face_quadrature_coord[0], P_face_quadrature_coord[1],
                    P_face_quadrature_coord[2]);

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  P_cell_rhs(i) += -time_step * EquationData::g_B_w *
                                   (fe_face_values.shape_value(i, q) *
                                    QP_bd_values[q] * fe_face_values.JxW(q));
                }
              }
            }
          }
        }
      }

      cell->get_dof_indices(P_local_dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          P_system_matrix.add(P_local_dof_indices[i], P_local_dof_indices[j],
                              P_cell_matrix(i, j));  // sys_mass matrix
        }
        P_system_rhs(P_local_dof_indices[i]) += P_cell_rhs(i);
      }
    }
  }

  P_system_matrix.compress(VectorOperation::add);
  P_system_rhs.compress(VectorOperation::add);

  {

    for (unsigned int bd_i = 0; bd_i < EquationData::g_num_P_bnd_id; ++bd_i) {

      P_boundary.get_bd_i(bd_i);
      P_boundary.set_time(time);
      P_boundary.set_boundary_id(*(EquationData::g_P_bnd_id + bd_i));
      std::map<types::global_dof_index, double> P_bd_values;
      VectorTools::interpolate_boundary_values(
          dof_handler, *(EquationData::g_P_bnd_id + bd_i), P_boundary,
          P_bd_values);  // i is boundary index
      LA::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
      MatrixTools::apply_boundary_values(P_bd_values, P_system_matrix, tmp,
                                         P_system_rhs, false);
      P_locally_relevant_solution = tmp;
    }
  }

  timer.tock("assemble_P_system");
}


template <int dim>
void CoupledTH<dim>::assemble_T_system() {
  cbgeo::Clock timer;
  timer.tick();
  T_system_rhs=0;
  T_system_matrix=0;
  T_locally_relevant_solution=0;

  // Getting fe values
  FEValues<dim> fe_values(fe_T, quadrature_formula_T,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FEValues<dim> fe_values_P(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(fe_T, face_quadrature_formula_T,
                                   update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

  FEValues<dim-2,dim> fe_line_values(fe_line, line_quadrature_formula,
                                   update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  // define loop number
  const unsigned int dofs_per_cell = fe_T.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula_T.size();
  const unsigned int n_face_q_points = face_quadrature_formula_T.size();
  const unsigned int dofs_per_line = fe_line.dofs_per_cell;
  const unsigned int n_q_points_line = line_quadrature_formula.size();

  // store the value at previous step at q_point for T
  Vector<double> old_T_sol_values_nodal(dofs_per_cell);

  // store the value at previous step at q_point for P
  std::vector<Tensor<1, dim>> old_P_sol_grads(n_q_points_P);

  // store the source and bd value at q_point
  std::vector<double> T_source_values(n_q_points);
  std::vector<double> QT_bd_values(n_face_q_points);

  //  local element matrix
  FullMatrix<double> T_cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> T_cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> T_local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> line_local_dof_indices(dofs_per_line);

  std::map<types::global_dof_index, double> old_Ttu;

  // boudnary condition
  EquationData::TemperatureSourceTerm<dim> T_source_term;
  EquationData::TemperatureNeumanBoundaryValues<dim> QT_boundary;
  EquationData::TemperatureDirichletBoundaryValues<dim> T_boundary;

  std::vector<types::global_dof_index> dof_appeared_local;

  // loop for cell
  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler_T.begin_active(),
                                                 endc = dof_handler_T.end();
  typename DoFHandler<dim>::active_cell_iterator cell_P =
                                                     dof_handler.begin_active(),
                                                 endc_P = dof_handler.end();

  for (; cell != endc; ++cell) {
    if (cell->is_locally_owned()) {
      cell->get_dof_indices(T_local_dof_indices);
      // initialization
      T_cell_matrix = 0;
      T_cell_rhs = 0;
      fe_values.reinit(cell);
      fe_values_P.reinit(cell_P);

      // get the values at gauss point old solution from the system
      if (time < 1e-8) {
        fe_values_P.get_function_gradients(initial_P_solution, old_P_sol_grads);
        //cell->get_dof_values(initial_T_solution, old_T_sol_values_nodal);
        old_T_sol_values_nodal = 273.15 + 18;
      } else {
        fe_values_P.get_function_gradients(old_P_locally_relevant_solution,
                                         old_P_sol_grads);
        cell->get_dof_values(old_T_locally_relevant_solution, old_T_sol_values_nodal);
      }

      // get right hand side
      T_source_term.set_time(time);
      T_source_term.value_list(fe_values.get_quadrature_points(),
                               T_source_values);

      // loop for q_point ASSMBLING CELL METRIX (weak form equation writing)
      for (unsigned int q = 0; q < n_q_points; ++q) {
        const auto T_quadrature_coord = fe_values.quadrature_point(q);
        const auto T_quadrature_coord_vector = fe_values_P.get_quadrature_points();
        auto it = find(T_quadrature_coord_vector.begin(), T_quadrature_coord_vector.end(), T_quadrature_coord);
        int index = it - T_quadrature_coord_vector.begin();

        // 3d interp
        EquationData::g_perm = data_interpolation.value(T_quadrature_coord[0],
                                                        T_quadrature_coord[1],
                                                        T_quadrature_coord[2]);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const Tensor<1, dim> grad_phi_i_T = fe_values.shape_grad(i, q);
          const double phi_i_T = fe_values.shape_value(i, q);
          const unsigned int component_i = fe_T.system_to_component_index(i).first;
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const Tensor<1, dim> grad_phi_j_T = fe_values.shape_grad(j, q);
            const double phi_j_T = fe_values.shape_value(j, q);
            const unsigned int component_j = fe_T.system_to_component_index(j).first;
            if (component_i==0 && component_j==0) {
              // mass matrix
              T_cell_matrix(i, j) += (phi_i_T * phi_j_T * fe_values.JxW(q));
              // stiff matrix
              T_cell_matrix(i, j) +=
                time_step * (EquationData::g_lam / EquationData::g_c_T *
                             grad_phi_i_T * grad_phi_j_T * fe_values.JxW(q));
              // conv matrix
              T_cell_matrix(i, j) +=
                time_step * EquationData::g_c_w / EquationData::g_c_T *
                phi_i_T *
                (-EquationData::g_perm *
                 (old_P_sol_grads[index] + (Point<dim>(0, 0, 1)) * EquationData::g_P_grad) *
                 grad_phi_j_T * fe_values.JxW(q));
              T_cell_rhs(i) += old_T_sol_values_nodal(j) * phi_j_T * phi_i_T * fe_values.JxW(q);
            }
          }
          if (component_i==0) {
            T_cell_rhs(i) += (time_step * T_source_values[q]) * phi_i_T * fe_values.JxW(q);
          }
          if (std::find(dof_appeared_Ttu.begin(), dof_appeared_Ttu.end(), T_local_dof_indices[i]) != dof_appeared_Ttu.end() &&
            std::find(dof_appeared_local.begin(), dof_appeared_local.end(), T_local_dof_indices[i]) == dof_appeared_local.end()){
            dof_appeared_local.push_back(T_local_dof_indices[i]);
            std::pair<types::global_dof_index, double> vd2 = std::make_pair(T_local_dof_indices[i],old_T_sol_values_nodal(i));
            old_Ttu.emplace(vd2);
          }
        }
      }

      // APPLIED NEUMAN BOUNDARY CONDITION

      for (unsigned int face_no = 0;
           face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {
        if (cell->at_boundary(face_no)) {
          for (int bd_i = 0; bd_i < EquationData::g_num_QT_bnd_id; ++bd_i) {
            if (cell->face(face_no)->boundary_id() ==
                EquationData::g_QT_bnd_id[bd_i]) {
              fe_face_values.reinit(cell, face_no);

              // get boundary condition
              QT_boundary.get_bd_i(bd_i);
              QT_boundary.set_time(time);
              QT_boundary.set_boundary_id(*(EquationData::g_QT_bnd_id + bd_i));
              QT_boundary.value_list(fe_face_values.get_quadrature_points(),
                                     QT_bd_values);

              for (unsigned int q = 0; q < n_face_q_points; ++q) {

                const auto T_face_quadrature_coord =
                    fe_face_values.quadrature_point(q);

                // 3d interp
                EquationData::g_perm = data_interpolation.value(
                    T_face_quadrature_coord[0], T_face_quadrature_coord[1],
                    T_face_quadrature_coord[2]);

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  const unsigned int component_i = fe_T.face_system_to_component_index(i,face_no).first;
                  if (component_i==0) {
                    T_cell_rhs(i) += -time_step / EquationData::g_c_T *
                                   fe_face_values.shape_value(i, q) *
                                   QT_bd_values[q] * fe_face_values.JxW(q);
                  }
                }
              }
            }
          }
        }
      }
      // local ->globe
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        const unsigned int component_i = fe_T.system_to_component_index(i).first;
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          const unsigned int component_j = fe_T.system_to_component_index(j).first;
          if (component_i==0 && component_j==0) {
            T_system_matrix.add(T_local_dof_indices[i], T_local_dof_indices[j],
                              T_cell_matrix(i, j));  //
          }
        }
        if (component_i==0) {
          T_system_rhs(T_local_dof_indices[i]) += T_cell_rhs(i);
        }
      }
    }
    ++cell_P;
  }

  double temp_u;
  if (time>172800) {
    temp_u=0;
  } else {
    temp_u=EquationData::u;
  }

  typename DoFHandler<dim-2,dim>::active_cell_iterator cell_line = dof_handler_line.begin_active(), endc_line = dof_handler_line.end();
  for (; cell_line != endc_line; ++cell_line) {
    cell_line->get_dof_indices(line_local_dof_indices);
    fe_line_values.reinit(cell_line);
    Point<3> &v0 = cell_line->vertex(fe_line.system_to_component_index(0).second);
    Point<3> &v1 = cell_line->vertex(fe_line.system_to_component_index(1).second);
    double h_ele = sqrt(pow(v0(0)-v1(0),2)+pow(v0(1)-v1(1),2)+pow(v0(2)-v1(2),2));
    double Pe = temp_u * h_ele / EquationData::k;
    double alpha = cosh(Pe/2)/sinh(Pe/2) - 2/Pe;

    for (unsigned int q = 0; q < n_q_points_line; ++q) {
      for (unsigned int i = 0; i < dofs_per_line; ++i) {
        const double phi_i_line2 = fe_line_values.shape_value(i, q);
        const Tensor<1, dim> grad_phi_i_line = fe_line_values.shape_grad(i, q);
        types::global_dof_index found_dof_i_Ts = dof_dof_map_Ts.at(line_local_dof_indices[i]);
        types::global_dof_index found_dof_i_Ttu = dof_dof_map_Ttu.at(line_local_dof_indices[i]);
        int formula_i;
        if (dof_dof_map_line.find(found_dof_i_Ttu)!=dof_dof_map_line.end()) {
          if (dof_dof_map_line.at(found_dof_i_Ttu)==dof_dof_map_Ttu.at(line_local_dof_indices[1-i])) {
            formula_i=1;
          } else {
            formula_i=2;
          }
        } else {
          formula_i=2;
        }
        const double phi_i_line = fe_line_values.shape_value(i, q)+pow(-1,formula_i)*alpha/2;

        for (unsigned int j = 0; j < dofs_per_line; ++j) {
          const double phi_j_line = fe_line_values.shape_value(j, q);
          const Tensor<1, dim> grad_phi_j_line = fe_line_values.shape_grad(j, q);
          types::global_dof_index found_dof_j_Ts = dof_dof_map_Ts.at(line_local_dof_indices[j]);
          types::global_dof_index found_dof_j_Ttu = dof_dof_map_Ttu.at(line_local_dof_indices[j]);
          types::global_dof_index found_dof_samez;
          types::global_dof_index found_dof_samez_Ttu;
          double temp_old_Ttu2;
          if (samez_dof_dof_map.find(line_local_dof_indices[j]) != samez_dof_dof_map.end()) {
            found_dof_samez = samez_dof_dof_map.at(line_local_dof_indices[j]);
            found_dof_samez_Ttu = dof_dof_map_Ttu.at(found_dof_samez);
            temp_old_Ttu2 = old_Ttu.at(found_dof_samez_Ttu);
          }
          double temp_old_Ttu = old_Ttu.at(found_dof_j_Ttu);
          //pcout<<temp_old_Ttu<<std::endl;
          double distance;
          
          if (dof_dof_map_line.find(found_dof_j_Ttu)!=dof_dof_map_line.end()) {
            if (dof_dof_map_line.at(found_dof_j_Ttu)==dof_dof_map_Ttu.at(line_local_dof_indices[1-j])) {
              distance=0;
            } else {
              distance=100;
            }
          } else {
            distance=100;
          }

          double hz_ff=EquationData::hz_ff;
          double bgs=EquationData::bgs;
          //double if_stop;
          //if (time>172800) {
          //  if_stop=0;
          //} else {
          //  if_stop=1;
          //}

          // mass matrix
          T_system_matrix.add(found_dof_i_Ts, found_dof_j_Ts, (phi_i_line2 * phi_j_line * bgs * time_step / EquationData::g_c_T * fe_line_values.JxW(q)));
          T_system_matrix.add(found_dof_i_Ts, found_dof_j_Ttu, (-phi_i_line2 * phi_j_line * bgs * time_step / EquationData::g_c_T * fe_line_values.JxW(q)));
          T_system_matrix.add(found_dof_i_Ttu, found_dof_j_Ts, (-time_step * hz_ff * phi_i_line * phi_j_line * fe_line_values.JxW(q)));
          double grad_phi_i_line_1d = sqrt(pow(grad_phi_i_line[0],2)+pow(grad_phi_i_line[1],2)+pow(grad_phi_i_line[2],2));
          double grad_phi_j_line_1d = sqrt(pow(grad_phi_j_line[0],2)+pow(grad_phi_j_line[1],2)+pow(grad_phi_j_line[2],2));
          
          if (distance<1) {
            T_system_matrix.add(found_dof_i_Ttu, found_dof_j_Ttu, ((time_step * hz_ff+EquationData::g_c_w * EquationData::A) * phi_i_line * phi_j_line * fe_line_values.JxW(q)
              + EquationData::g_c_w * EquationData::A * time_step * temp_u * phi_i_line * (-grad_phi_j_line_1d) * fe_line_values.JxW(q)
              + time_step * EquationData::A * EquationData::k * grad_phi_i_line * grad_phi_j_line * fe_line_values.JxW(q)));
          } else {
            T_system_matrix.add(found_dof_i_Ttu, found_dof_j_Ttu, ((time_step * hz_ff+EquationData::g_c_w * EquationData::A) * phi_i_line * phi_j_line * fe_line_values.JxW(q)
              + EquationData::g_c_w * EquationData::A * time_step * temp_u * phi_i_line * (grad_phi_j_line_1d) * fe_line_values.JxW(q)
              + time_step * EquationData::A * EquationData::k * grad_phi_i_line * grad_phi_j_line * fe_line_values.JxW(q)));
          }
          if (samez_dof_dof_map.find(line_local_dof_indices[j]) != samez_dof_dof_map.end()) {
            //T_system_matrix.add(found_dof_i_Ttu, found_dof_j_Ttu, time_step * EquationData::lambda_tube * phi_i_line * phi_j_line * fe_line_values.JxW(q));
            //T_system_matrix.add(found_dof_i_Ttu, found_dof_samez_Ttu, -time_step * EquationData::lambda_tube * phi_i_line * phi_j_line * fe_line_values.JxW(q));
            T_system_rhs(found_dof_i_Ttu) += (-temp_old_Ttu+temp_old_Ttu2) * EquationData::lambda_tube * phi_i_line * phi_j_line * fe_line_values.JxW(q);
          }
          //T_system_rhs(found_dof_i_Ttu) += 2*(EquationData::g_c_w * EquationData::A * phi_i_line * phi_j_line * temp_old_Ttu * fe_line_values.JxW(q));
          T_system_rhs(found_dof_i_Ttu) += (EquationData::g_c_w * EquationData::A * phi_i_line * phi_j_line * temp_old_Ttu * fe_line_values.JxW(q) +
            time_step * EquationData::fD * EquationData::rho * EquationData::A / (2*EquationData::d) * pow(temp_u,3) * phi_i_line * fe_line_values.JxW(q));
        }
      }
    }
  }

  // compress the matrix
  T_system_matrix.compress(VectorOperation::add);
  T_system_rhs.compress(VectorOperation::add);

  // ADD DIRICHLET BOUNDARY
  {
    std::map<types::global_dof_index, double> T_bd_values;
    LA::MPI::Vector tmp(locally_owned_dofs_T, mpi_communicator);

    for (int bd_i = 0; bd_i < EquationData::g_num_T_bnd_id; bd_i++) {
      T_boundary.get_bd_i(bd_i);
      T_boundary.get_period(period);  // for seeting sin function
      T_boundary.set_time(time);
      T_boundary.set_boundary_id(*(EquationData::g_T_bnd_id + bd_i));
      

      VectorTools::interpolate_boundary_values(
          dof_handler_T, *(EquationData::g_T_bnd_id + bd_i), T_boundary,
          T_bd_values);  // i is boundary index
    }

    dummy_map.insert(T_bd_values.begin(), T_bd_values.end());
    MatrixTools::apply_boundary_values(dummy_map, T_system_matrix, tmp,
                                         T_system_rhs, false);
    T_locally_relevant_solution = tmp;
  }

  timer.tock("assemble_T_system");
}

template <int dim>
void CoupledTH<dim>::linear_solve_P() {
  cbgeo::Clock timer;
  timer.tick();

  LA::MPI::Vector distributed_P_solution(locally_owned_dofs, mpi_communicator);
  SolverControl solver_control(
      std::max<std::size_t>(n_P_max_iteration, P_system_rhs.size()),
      P_tol_residual * P_system_rhs.l2_norm());  // setting for solver
  // pcout<< "\n the l1 norm of the P_system is"<< P_system_matrix.l1_norm() <<
  // "\n";
  distributed_P_solution = P_locally_relevant_solution;

  // LA::SolverGMRES solver(solver_control, mpi_communicator);  // config cg
  LA::SolverCG solver(solver_control, mpi_communicator);  // config cg
  // LA::MPI::PreconditionJacobi preconditioner(P_system_matrix);
  PETScWrappers::PreconditionBlockJacobi preconditioner(P_system_matrix);
  solver.solve(P_system_matrix, distributed_P_solution, P_system_rhs,
               preconditioner);  // solve eq

  P_locally_relevant_solution = distributed_P_solution;

  old_P_locally_relevant_solution = distributed_P_solution;

  P_iteration_namber = solver_control.last_step();

  timer.tock("linear_solve_P");
  pcout << "\nIterations required for P convergence: " << P_iteration_namber
        << "\n";
}

template <int dim>
void CoupledTH<dim>::linear_solve_T() {
  cbgeo::Clock timer;
  timer.tick();
  LA::MPI::Vector distributed_T_solution(locally_owned_dofs_T, mpi_communicator);

  SolverControl solver_control(
      100000,
      //std::max<std::size_t>(n_T_max_iteration, T_system_rhs.size()),
      T_tol_residual * T_system_rhs.l2_norm(), true, true);  // setting for solver
  // pcout<< "\n the l1 norm of the T_system is"<< T_system_matrix.l1_norm() <<
  //   "\n";

  distributed_T_solution = T_locally_relevant_solution;
  LA::SolverGMRES solver(solver_control,
                         mpi_communicator);  // config solver
  // LA::MPI::PreconditionAMG preconditioner;
  // LA::MPI::PreconditionAMG::AdditionalData data;
  // data.symmetric_operator = false;
  // preconditioner.initialize(T_system_matrix, data);

  LA::MPI::PreconditionJacobi preconditioner(T_system_matrix);  // precond
  // preconditioner.initialize(T_system_matrix, 1.0);      // initialize precond
  solver.solve(T_system_matrix, distributed_T_solution, T_system_rhs,
               preconditioner);  // solve eq

  // T_constraints.distribute(distributed_T_solution);
  T_locally_relevant_solution = distributed_T_solution;

  old_T_locally_relevant_solution = distributed_T_solution;

  T_iteration_namber = solver_control.last_step();

  timer.tock("linear_solve_T");

  pcout << " \nIterations required for T convergence:    " << T_iteration_namber
        << "\n";
}

// @sect4{<code>CoupledTH::output_results</code>}
//
// Neither is there anything new in generating graphical output:
template <int dim>
void CoupledTH<dim>::output_results(LA::MPI::Vector& solution,
                                    std::string var_name) const {

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);

  data_out.add_data_vector(solution, var_name);

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  const std::string filename =
      var_name + "-solution-" + Utilities::int_to_string(timestep_number, 3);

  const std::string pvtu_master_filename = data_out.write_vtu_with_pvtu_record(
      "outputfiles3/", filename, timestep_number, mpi_communicator);

  if (this_mpi_process == 0) {
    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.push_back(
        std::pair<double, std::string>(time, pvtu_master_filename));
    std::ofstream pvd_output("outputfiles3/" + var_name + "_solution.pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }
}

template <int dim>
void CoupledTH<dim>::output_results2(LA::MPI::Vector& solution,
                                    std::string var_name) const {

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler_T);

  std::vector<std::string> solution_names;
  solution_names.emplace_back("Ts");
  solution_names.emplace_back("Ttu");
  data_out.add_data_vector(solution, solution_names);

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  const std::string filename =
      var_name + "-solution-" + Utilities::int_to_string(timestep_number, 3);

  const std::string pvtu_master_filename = data_out.write_vtu_with_pvtu_record(
      "outputfiles3/", filename, timestep_number, mpi_communicator);

  if (this_mpi_process == 0) {
    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.push_back(
        std::pair<double, std::string>(time, pvtu_master_filename));
    std::ofstream pvd_output("outputfiles3/" + var_name + "_solution.pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }
}

template <int dim>
void CoupledTH<dim>::run() {
  cbgeo::Clock timer;
  timer.tick();

  unsigned int binary_search_number;
  double initial_time_step;
  double theta;

  make_grid_and_dofs();

  setup_P_system();
  setup_T_system();

  VectorTools::interpolate(dof_handler_T,
                           EquationData::TemperatureInitialValues<dim>(),
                           initial_T_solution);
  VectorTools::interpolate(dof_handler,
                           EquationData::PressureInitialValues<dim>(),
                           initial_P_solution);

  do {

    pcout << "\nTimestep " << timestep_number;

    binary_search_number = 1;
    initial_time_step =
        time_sequence[timestep_number + 1] - time_sequence[timestep_number];
    time_step = initial_time_step / 2;
    theta = 0;

    do {

      assemble_P_system();

      linear_solve_P();

      assemble_T_system();

      linear_solve_T();

      time += time_step;

      theta += pow(0.5, binary_search_number);

      if (P_iteration_namber > n_P_max_iteration / 2 ||
          T_iteration_namber > n_T_max_iteration / 2) {
        time_step = time_step / 2;
        ++binary_search_number;
      }

      pcout << "   \n theta  = " << theta << std::endl;

    } while ((1 - theta) > 0.00001);

    pcout << "   \n Solver converged in " << binary_search_number
          << " time divisions." << std::endl;

    timestep_number += 1;

    pcout << "\nt=" << time << ", dt=" << time_step << '.' << std::endl;

    output_results2(T_locally_relevant_solution, "T");
    output_results(P_locally_relevant_solution, "P");

    pcout << "\n" << std::endl << std::endl;

    // MatrixOut matrix_out;
    // std::ofstream out_T_matrix
    // ("/outputfiles/2rhs_T_matrix_at_"+std::to_string(time));
    // matrix_out.build_patches (T_system_matrix, "T_system_matrix");
    // matrix_out.write_gnuplot (out_T_matrix);
    // T_system_matrix.print_formatted(out_T_matrix);
    // T_system_rhs.print(out);

  } while (time < total_time);

  timer.tock("solve_all");
  pcout << "\n" << std::endl << std::endl;
}
