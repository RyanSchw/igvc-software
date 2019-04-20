#include <cmath>

/**
 * Templated interface the Some controller.
 *
 * Author: Oswin So <oswinso@gmail.com>
 * Date Created: April 13th, 2019
 */
#ifndef SRC_SOME_CONTROLLER_H
#define SRC_SOME_CONTROLLER_H

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "cost_function.h"
#include "model.h"

namespace some_controller
{
template <class ModelImpl>
struct Particle
{
  std::vector<typename ModelImpl::Controls> controls_vec_;
  std::vector<typename ModelImpl::StateType> state_vec_;
  std::vector<float> cum_cost_;

public:
  void initialize(const typename ModelImpl::StateType& initial_state, int iterations);
  typename ModelImpl::StateType getState() const;
  float getWeight() const;
  template <class M>
  friend std::ostream& operator<<(std::ostream& out, const Particle<M>& particle);
};

template <class Model>
void Particle<Model>::initialize(const typename Model::StateType& initial_state, int iterations)
{
  controls_vec_.clear();
  state_vec_.clear();
  cum_cost_.clear();

  state_vec_.reserve(iterations + 1);
  cum_cost_.reserve(iterations + 1);
  state_vec_.reserve(iterations);

  state_vec_.emplace_back(initial_state);
  cum_cost_.emplace_back(0.0f);
}

template <class ModelImpl>
typename ModelImpl::StateType Particle<ModelImpl>::getState() const
{
  return state_vec_.back();
}

template <class Model>
float Particle<Model>::getWeight() const
{
  return 1 / (cum_cost_.back() * cum_cost_.back());
}

template <class ModelImpl>
struct OptimizationResult
{
  std::vector<Particle<ModelImpl>> particles;
  Particle<ModelImpl> weighted_particle;

public:
  OptimizationResult(std::vector<Particle<ModelImpl>>&& moved_particles, Particle<ModelImpl>&& moved_weighted) noexcept;
};

struct SomeControllerOptions
{
  float timestep;
  float horizon;
  int num_samples;
  float resample_threshold;
  float std_dev;
  bool use_weighted;
  float last_state_coeff;
};

template <class ModelImpl, class CostFunctionImpl>
class SomeController
{
public:
  using Controls = typename ModelImpl::Controls;
  using State = typename ModelImpl::StateType;
  constexpr static const int control_dims = ModelImpl::control_dims;
  using CRTPModel = Model<State, control_dims, ModelImpl>;
  using CRTPCostFunction = CostFunction<State, control_dims, CostFunctionImpl>;

  SomeController(std::shared_ptr<Model<State, control_dims, ModelImpl>> model,
                 std::shared_ptr<CostFunction<State, control_dims, CostFunctionImpl>> cost_function,
                 const SomeControllerOptions& options);
  std::unique_ptr<OptimizationResult<ModelImpl>> optimize(const State& starting_state);

private:
  void resampleParticles();
  float effectiveParticles();
  void initializeParticles(const State& starting_state);
  std::vector<Controls> getWeightedPath();
  Particle<ModelImpl> simulateWithWeightedControls(const State& initial_state,
                                                   const std::vector<Controls>& controls) const;
  Controls sampleControls() const;
  void setControls(const std::vector<Controls>& controls);

  std::shared_ptr<Model<State, control_dims, ModelImpl>> model_;
  std::shared_ptr<CostFunction<State, control_dims, CostFunctionImpl>> cost_function_;
  float timestep_;
  float horizon_;
  int iterations_;
  int num_samples_;
  float resample_threshold_;
  bool use_weighted_;
  float std_dev_;
  float last_state_coeff_;

  std::random_device rd_;
  mutable std::mt19937 mt_;
  std::array<std::normal_distribution<float>, control_dims> distributions_;

  std::vector<Particle<ModelImpl>> particles_;
};

template <class ModelImpl, class CostFunctionImpl>
SomeController<ModelImpl, CostFunctionImpl>::SomeController(
    std::shared_ptr<Model<State, control_dims, ModelImpl>> model,
    std::shared_ptr<CostFunction<State, control_dims, CostFunctionImpl>> cost_function,
    const SomeControllerOptions& options)
  : model_{ model }
  , cost_function_{ cost_function }
  , timestep_{ options.timestep }
  , horizon_{ options.horizon }
  , iterations_{ static_cast<int>(std::round(options.horizon / options.timestep)) * 10 }
  , num_samples_{ options.num_samples }
  , resample_threshold_{ options.resample_threshold }
  , use_weighted_{ options.use_weighted }
  , rd_{}
  , mt_{ rd_() }
  , particles_(options.num_samples, Particle<ModelImpl>{})
  , std_dev_(options.std_dev)
  , last_state_coeff_(options.last_state_coeff)
{
  ROS_INFO_STREAM("Constructor called! particles_.size(): " << particles_.size());
  std::array<Bound, control_dims> bounds = model_->bounds();
  for (int i = 0; i < control_dims; i++)
  {
    Bound bound = bounds[i];
    distributions_[i] = std::normal_distribution<float>(0.0f, std_dev_);
    ROS_INFO_STREAM("min: " << distributions_[i].min() << ", max: " << distributions_[i].max());
  }
}

template <class T>
OptimizationResult<T>::OptimizationResult(std::vector<Particle<T>>&& moved_particles,
                                          Particle<T>&& moved_weighted) noexcept
  : particles(std::move(moved_particles)), weighted_particle(std::move(moved_weighted))
{
}

template <class ModelImpl, class CostFunctionImpl>
std::unique_ptr<OptimizationResult<ModelImpl>>
SomeController<ModelImpl, CostFunctionImpl>::optimize(const State& starting_state)
{
  initializeParticles(starting_state);

  for (int i = 0; i < iterations_ / 10; i++)
  {
    for (Particle<ModelImpl>& particle : particles_)
    {
      setControls(particle.controls_vec_);
      Controls controls = sampleControls();

      for (int j = 0; j < 10; j++) {
        State state = particle.getState();
        State new_state = model_->doPropogateState(state, controls, timestep_ / 10);
        float cost = cost_function_->cost(new_state, controls);

        particle.cum_cost_.emplace_back(particle.cum_cost_.back() + cost);
        particle.controls_vec_.emplace_back(std::move(controls));
        particle.state_vec_.emplace_back(std::move(new_state));
      }
    }
    if (effectiveParticles() < resample_threshold_) {
//      ROS_INFO_STREAM("YES resampling! effective particles: " << effectiveParticles());
      resampleParticles();
    } else {
//      ROS_INFO_STREAM("NO resampling! effective particles: " << effectiveParticles());
    }
  }
  for (Particle<ModelImpl>& particle : particles_)
  {
    State state = particle.getState();
    float cost = cost_function_->cost(state, particle.controls_vec_.back());

    particle.cum_cost_.back() = particle.cum_cost_[particle.cum_cost_.size() - 1] + last_state_coeff_ * cost;
  }

  Particle<ModelImpl> best;
  if (use_weighted_) {
    std::vector<Controls> weighted_controls = getWeightedPath();
    best = simulateWithWeightedControls(starting_state, weighted_controls);
  } else {
    auto optimal_particle = std::min_element(particles_.begin(), particles_.end(),
      [](const Particle<ModelImpl>& p1, const Particle<ModelImpl>& p2) {
        return p1.cum_cost_.back() < p2.cum_cost_.back();
      });
    std::vector<Controls> optimal_controls = optimal_particle->controls_vec_;
    best = simulateWithWeightedControls(starting_state, optimal_controls);
  }
  return std::make_unique<OptimizationResult<ModelImpl>>(std::move(particles_), std::move(best));
}

template <class ModelImpl, class CostFunctionImpl>
float SomeController<ModelImpl, CostFunctionImpl>::effectiveParticles() {
  float sum = 0;
  float squared_normalized_sum = 0;
  auto worst_particle =
    std::min_element(particles_.begin(), particles_.end(), [](const Particle<ModelImpl>& p1, const Particle<ModelImpl>& p2) {
      return p1.cum_cost_.back() > p2.cum_cost_.back();
    });
  for (const Particle<ModelImpl>& particle : particles_) {
    sum += particle.getWeight() - worst_particle->getWeight();
  }
  for (const Particle<ModelImpl>& particle : particles_) {
    squared_normalized_sum += ((particle.getWeight() - worst_particle->getWeight()) * (particle.getWeight() - worst_particle->getWeight())) / (sum * sum);
  }
  return 1 / (squared_normalized_sum * num_samples_);
}


template <class ModelImpl, class CostFunctionImpl>
Particle<ModelImpl> SomeController<ModelImpl, CostFunctionImpl>::simulateWithWeightedControls(
    const State& initial_state, const std::vector<Controls>& controls) const
{
  Particle<ModelImpl> particle{};
  particle.initialize(initial_state, iterations_);

  particle.controls_vec_ = controls;
  for (const Controls& control : controls)
  {
    State state = particle.getState();
    State new_state = model_->doPropogateState(state, control, timestep_ / 10);
    float cost = cost_function_->cost(new_state, control);
    particle.cum_cost_.emplace_back(particle.cum_cost_.back() + cost);
    particle.state_vec_.emplace_back(std::move(new_state));
  }
  return particle;
}

template <class ModelImpl, class CostFunctionImpl>
std::vector<typename ModelImpl::Controls> SomeController<ModelImpl, CostFunctionImpl>::getWeightedPath()
{
  std::vector<Controls> weighted_controls(iterations_, Controls{});
  float cum_weight = 0;
  for (const Particle<ModelImpl>& particle : particles_)
  {
    float weight = particle.getWeight();
    cum_weight += weight;
    for (int i = 0; i < iterations_; i++)
    {
      for (int j = 0; j < control_dims; j++)
      {
        weighted_controls[i][j] += weight * particle.controls_vec_[i][j];
      }
    }
  }

  for (Controls& control : weighted_controls)
  {
    for (float& ind_control : control)
    {
      ind_control = ind_control / cum_weight;
    }
  }
  return weighted_controls;
}

template <class ModelImpl, class CostFunctionImpl>
typename ModelImpl::Controls SomeController<ModelImpl, CostFunctionImpl>::sampleControls() const
{
  Controls controls;
  for (int i = 0; i < control_dims; i++)
  {
    std::normal_distribution distribution = distributions_[i];
    controls[i] = distribution(mt_);
  }
  return controls;
}

template <class ModelImpl, class CostFunctionImpl>
void SomeController<ModelImpl, CostFunctionImpl>::setControls(const std::vector<typename ModelImpl::Controls>& controls)
{
  if (!controls.empty()) {
    for (int i = 0; i < control_dims; i++) {
      distributions_[i] = std::normal_distribution<float>(controls.back()[i], std_dev_);
    }
  } else {
    for (int i = 0; i < control_dims; i++) {
      distributions_[i] = std::normal_distribution<float>(0, std_dev_);
    }
  }
}

template <class ModelImpl, class CostFunctionImpl>
void SomeController<ModelImpl, CostFunctionImpl>::initializeParticles(const State& starting_state)
{
  particles_ = std::vector<Particle<ModelImpl>>(num_samples_, Particle<ModelImpl>{});
  for (Particle<ModelImpl>& particle : particles_)
  {
    particle.initialize(starting_state, iterations_);
  }
}

template <class ModelImpl, class CostFunctionImpl>
void SomeController<ModelImpl, CostFunctionImpl>::resampleParticles()
{
  std::vector<float> cum_weights;
  cum_weights.reserve(static_cast<unsigned long>(num_samples_));
  auto worst_particle =
    std::min_element(particles_.begin(), particles_.end(), [](const Particle<ModelImpl>& p1, const Particle<ModelImpl>& p2) {
      return p1.cum_cost_.back() > p2.cum_cost_.back();
    });
  cum_weights.emplace_back(particles_.front().getWeight() - worst_particle->getWeight());
  for (int i = 1; i < num_samples_; i++)
  {
    cum_weights.emplace_back(cum_weights.back() + particles_[i].getWeight() - worst_particle->getWeight());
  }
  float pointer_width = cum_weights.back() / num_samples_;
  std::uniform_real_distribution<float> unif(0, pointer_width);
  float starting_pointer = unif(mt_);

  std::vector<Particle<ModelImpl>> sampled_particles;
  sampled_particles.reserve(static_cast<unsigned long>(num_samples_));
  for (int i = 0; i < num_samples_; i++)
  {
    int index = 0;
    while (cum_weights[index] < starting_pointer + i * pointer_width)
    {
      index++;
    }
    sampled_particles.emplace_back(particles_[index]);
  }
  particles_ = std::move(sampled_particles);
}

template <class ModelImpl>
std::ostream& operator<<(std::ostream& out, const Particle<ModelImpl>& particle)
{
  out << std::setprecision(3) << "State"
      << "\t\t"
      << "Control"
      << "\t\t"
      << "Cost" << std::endl;
  for (size_t i = 0; i < particle.controls_vec_.size(); i++)
  {
    out << std::setprecision(3) << particle.state_vec_[i + 1] << "\t\t"
        << "[";
    std::copy(particle.controls_vec_[i].begin(), particle.controls_vec_[i].end(),
              std::ostream_iterator<float>(out, ", "));
    out << "]"
        << "\t\t" << particle.cum_cost_[i + 1] << std::endl;
  }
  out << std::endl;
  return out;
}

}  // namespace some_controller

#endif  // SRC_SOME_CONTROLLER_H
