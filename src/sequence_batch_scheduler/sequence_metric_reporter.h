// Copyright 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include "status.h"
#include "triton/common/model_config.h"

#ifdef TRITON_ENABLE_METRICS
#include "metrics.h"
#include "model.h"
#include "prometheus/registry.h"
#endif  // TRITON_ENABLE_METRICS

namespace triton { namespace core {

#ifdef TRITON_ENABLE_METRICS
struct ModelIdentifier;
#endif  // TRITON_ENABLE_METRICS

//
// Interface for a metric reporter for a given version of a model.
//
class SequenceMetricReporter {
 public:
#ifdef TRITON_ENABLE_METRICS
  static Status Create(
      const triton::core::ModelIdentifier& model_id,
      const int64_t model_version,
      const triton::common::MetricTagsMap& model_tags,
      std::shared_ptr<SequenceMetricReporter>* sequence_metric_reporter);

  ~SequenceMetricReporter();

  // Lookup counter metric by name, and increment it by value if it exists.
  void IncrementCounter(const std::string& name, double value);
  // Increase gauge by value.
  void IncrementGauge(const std::string& name, double value);
  // Decrease gauge by value.
  void DecrementGauge(const std::string& name, double value);
  // Lookup summary metric by name, and observe the value if it exists.
  void ObserveSummary(const std::string& name, double value);

 private:
 SequenceMetricReporter(
      const ModelIdentifier& model_id, const int64_t model_version,
      const triton::common::MetricTagsMap& model_tags);

  static void GetMetricLabels(
      std::map<std::string, std::string>* labels,
      const ModelIdentifier& model_id, const int64_t model_version,
      const triton::common::MetricTagsMap& model_tags);

  template <typename T, typename... Args>
  T* CreateMetric(
      prometheus::Family<T>& family,
      const std::map<std::string, std::string>& labels, Args&&... args);

  void InitializeMetrics(const std::map<std::string, std::string>& labels);

  // Lookup gauge metric by name. Return gauge if found, nullptr otherwise.
  prometheus::Gauge* GetGauge(const std::string& name);


  // Metric Families
  std::unordered_map<std::string, prometheus::Family<prometheus::Counter>*>
      counter_families_;
  std::unordered_map<std::string, prometheus::Family<prometheus::Gauge>*>
      gauge_families_;

  // Metrics
  std::unordered_map<std::string, prometheus::Counter*> counters_;
  std::unordered_map<std::string, prometheus::Gauge*> gauges_;
  bool sequence_metrics_enabled_;
#endif  // TRITON_ENABLE_METRICS
};

}}  // namespace triton::core
