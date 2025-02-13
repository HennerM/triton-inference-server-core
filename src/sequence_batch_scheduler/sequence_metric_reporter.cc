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

#include "sequence_metric_reporter.h"

#ifdef TRITON_ENABLE_METRICS

#include "constants.h"

namespace triton { namespace core {

Status
SequenceMetricReporter::Create(
    const ModelIdentifier& model_id, const int64_t model_version,
    const triton::common::MetricTagsMap& model_tags,
    std::shared_ptr<SequenceMetricReporter>* sequence_metric_reporter)
{
  static std::mutex mtx;
  static std::unordered_map<size_t, std::weak_ptr<SequenceMetricReporter>>
      reporter_map;

  std::map<std::string, std::string> labels;
  GetMetricLabels(&labels, model_id, model_version, model_tags);
  auto hash_labels = Metrics::HashLabels(labels);

  std::lock_guard<std::mutex> lock(mtx);

  const auto& itr = reporter_map.find(hash_labels);
  if (itr != reporter_map.end()) {
    // Found in map. If the weak_ptr is still valid that means that
    // there are other models using the reporter and we just reuse that
    // same reporter. If the weak_ptr is not valid then we need to remove
    // the weak_ptr from the map and create the reporter again.
    *sequence_metric_reporter = itr->second.lock();
    if (*sequence_metric_reporter != nullptr) {
      return Status::Success;
    }

    reporter_map.erase(itr);
  }

  sequence_metric_reporter->reset(new SequenceMetricReporter(
      model_id, model_version, model_tags));
  reporter_map.insert({hash_labels, *sequence_metric_reporter});
  return Status::Success;
}

SequenceMetricReporter::SequenceMetricReporter(
    const ModelIdentifier& model_id, const int64_t model_version,
    const triton::common::MetricTagsMap& model_tags)
{
  std::map<std::string, std::string> labels;
  GetMetricLabels(&labels, model_id, model_version, model_tags);

  // Initialize families and metrics
  InitializeMetrics(labels);
}

SequenceMetricReporter::~SequenceMetricReporter()
{
  // Cleanup metrics for each family
  for (auto& iter : counter_families_) {
    const auto& name = iter.first;
    auto family_ptr = iter.second;
    if (family_ptr) {
      family_ptr->Remove(counters_[name]);
    }
  }

  for (auto& iter : gauge_families_) {
    const auto& name = iter.first;
    auto family_ptr = iter.second;
    if (family_ptr) {
      family_ptr->Remove(gauges_[name]);
    }
  }

}

void
SequenceMetricReporter::InitializeMetrics(
  const std::map<std::string, std::string>& labels)
{
  if (sequence_metrics_enabled_) {
    
    counter_families_[kSequenceBacklogSequencesQueuedMetric] =
        &Metrics::FamilySequenceBacklogSequencesQueued();
    counter_families_[kSequenceBacklogRequestsQueuedMetric] =
        &Metrics::FamilySequenceBacklogRequestsQueued();
    counter_families_[kSequenceBacklogExpiredMetric] =
        &Metrics::FamilySequenceBacklogExpired();
    counter_families_[kSequenceBacklogCancelledMetric] = 
        &Metrics::FamilySequenceBacklogCancelled();

    gauge_families_[kSequenceBacklogSequencesMetric] =
        &Metrics::FamilySequenceBacklogSequences();
    gauge_families_[kSequenceBacklogRequestsMetric] =
        &Metrics::FamilySequenceBacklogRequests();

    for (auto& iter : gauge_families_) {
      const auto& name = iter.first;
      auto family_ptr = iter.second;
      if (family_ptr) {
        gauges_[name] = CreateMetric<prometheus::Gauge>(*family_ptr, labels);
      }
    }

    for (auto& iter : counter_families_) {
      const auto& name = iter.first;
      auto family_ptr = iter.second;
      if (family_ptr) {
        counters_[name] = CreateMetric<prometheus::Counter>(*family_ptr, labels);
      }
    }
  }
}

void
SequenceMetricReporter::GetMetricLabels(
    std::map<std::string, std::string>* labels, const ModelIdentifier& model_id,
    const int64_t model_version,
    const triton::common::MetricTagsMap& model_tags)
{
  if (!model_id.NamespaceDisabled()) {
    labels->insert(std::map<std::string, std::string>::value_type(
        std::string(kMetricsLabelModelNamespace), model_id.namespace_));
  }
  labels->insert(std::map<std::string, std::string>::value_type(
      std::string(kMetricsLabelModelName), model_id.name_));
  labels->insert(std::map<std::string, std::string>::value_type(
      std::string(kMetricsLabelModelVersion), std::to_string(model_version)));
  for (const auto& tag : model_tags) {
    labels->insert(std::map<std::string, std::string>::value_type(
        "_" + tag.first, tag.second));
  }

}

template <typename T, typename... Args>
T*
SequenceMetricReporter::CreateMetric(
    prometheus::Family<T>& family,
    const std::map<std::string, std::string>& labels, Args&&... args)
{
  return &family.Add(labels, args...);
}


void
SequenceMetricReporter::IncrementCounter(const std::string& name, double value)
{
  auto iter = counters_.find(name);
  if (iter == counters_.end()) {
    // No counter metric exists with this name
    return;
  }

  auto counter = iter->second;
  if (!counter) {
    // Counter is uninitialized/nullptr
    return;
  }
  counter->Increment(value);
}

prometheus::Gauge*
SequenceMetricReporter::GetGauge(const std::string& name)
{
  auto iter = gauges_.find(name);
  if (iter == gauges_.end()) {
    // No gauge metric exists with this name
    return nullptr;
  }

  auto gauge = iter->second;
  return gauge;
}

void
SequenceMetricReporter::IncrementGauge(const std::string& name, double value)
{
  auto gauge = GetGauge(name);
  if (gauge) {
    gauge->Increment(value);
  }
}

void
SequenceMetricReporter::DecrementGauge(const std::string& name, double value)
{
  IncrementGauge(name, -1 * value);
}

}}  // namespace triton::core

#endif  // TRITON_ENABLE_METRICS
