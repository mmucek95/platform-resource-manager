# Copyright (C) 2019 Intel Corporation
#  
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  
# http://www.apache.org/licenses/LICENSE-2.0
#  
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: Apache-2.0


import os
import logging

from prm.accuracy import (build_prometheus_url, fetch_metrics,
                          calculate_components, calculate_precision_and_recall)

from time import time
import requests


def _get_mesos_running_tasks(mesos_master_host):
    tasks_response = requests.post(
        'http://%s:5050/api/v1' % mesos_master_host,
        data='{"type": "GET_TASKS"}',
        headers={'content-type': 'application/json'}
    )
    tasks_response.raise_for_status()
    tasks = tasks_response.json()
    if 'tasks' not in tasks['get_tasks']:
        return []
    else:
        return sorted([t['name'] for t in tasks['get_tasks']['tasks']])


def test_integration_accurracy(record_property):
    """ Integration tests to check number of runnings tasks during scenario
    and calculate and output them to csv file for visualization. """
    assert 'MESOS_MASTER_HOST' in os.environ, 'required to get number of running tasks'
    assert 'MESOS_EXPECTED_TASKS' in os.environ, 'required to check number of tasks running'
    assert 'PROMETHEUS' in os.environ, 'prometheus host to connect'
    assert 'BUILD_NUMBER' in os.environ
    assert 'BUILD_COMMIT' in os.environ
    assert 'BUILD_SCENARIO' in os.environ
    assert 'MIN_RECALL' in os.environ
    assert 'MIN_PRECISION' in os.environ

    mesos_master_host = os.environ['MESOS_MASTER_HOST']
    prometheus = os.environ['PROMETHEUS']
    build_number = int(os.environ['BUILD_NUMBER'])
    build_commit = os.environ['BUILD_COMMIT']
    build_scenario = os.environ['BUILD_SCENARIO']
    tags = dict(build_number=build_number,
                build_scenario=build_scenario,
                build_commit=build_commit)
    mesos_expected_tasks = int(os.environ['MESOS_EXPECTED_TASKS'])
    window_size = float(os.environ.get('WINDOW_SIZE', 10.0))
    min_recall = float(os.environ.get('MIN_RECALL', -1))
    min_precision = float(os.environ.get('MIN_PRECISION', -1))

    logging.info('window size = %s', window_size)
    logging.info('build number = %r', build_number)
    logging.info('min recall = %s', min_recall)
    logging.info('min precision = %r', min_precision)

    # Check running tasks.
    tasks = _get_mesos_running_tasks(mesos_master_host)
    logging.info('tasks = %s', len(tasks))
    assert len(tasks) >= mesos_expected_tasks, \
        'invalid number of tasks: %r (expected=%r)' % (
            len(tasks), mesos_expected_tasks)

    # Calculate results.
    prometheus_anomalies_query = build_prometheus_url(prometheus, 'anomaly',
                                                      tags, 3600, time())
    logging.debug('prometheus query = %r', prometheus_anomalies_query)

    anomalies = fetch_metrics(prometheus_anomalies_query)
    if anomalies['data']['result']:
        logging.info('found anomalies = %s', len(anomalies['data']['result']))
    else:
        logging.info('No anomalies found.')
        return

    true_positives, anomaly_count, slo_violations = calculate_components(
        anomalies, prometheus, tags, window_size)
    logging.debug('found true positives = %s', true_positives)
    logging.debug('found anomaly count = %s', anomaly_count)
    logging.debug('found slo violations count = %s', slo_violations)

    precision, recall = calculate_precision_and_recall(
        true_positives, anomaly_count, slo_violations)

    record_property('build_number', build_number)
    record_property('recall', recall)
    record_property('precision', precision)
    record_property('tasks', tasks)

    logging.info('recall = %s', recall)
    logging.info('precision = %s', precision)

    with open('test_results.csv', 'w') as f:
        f.write('recall,precision,tasks,anomaly_count,slo_violations\n')
        f.write('%s,%s,%s,%s,%s\n' % (
            recall, precision, len(tasks), anomaly_count, slo_violations))

    assert precision >= min_precision, 'Excepted to get at least %s' % min_precision
    assert recall >= min_recall, 'Expected to get at least %s' % min_recall
