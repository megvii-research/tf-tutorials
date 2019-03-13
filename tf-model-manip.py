#!/usr/bin/env mdl
# -*- coding: utf-8 -*-
# =======================================
# File Name :
# Purpose :
# Creation Date :
# Last Modified :
# Created By : sunpeiqin
# =======================================

import os
import sys
import argparse
import magic
import keyword
import importlib
import collections
import re
import tabulate
import numpy as np

import tensorflow as tf


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "{:3.3f} {}{}".format(num, unit, suffix)
        num /= 1024.0
    sign_str = '-' if num < 0 else ''
    return "{}{:.1f} {}{}".format(sign_str, num, 'Yi', suffix)

def import_python_source_as_module(fpath, mod_name=None):
    """ import a python source as a module; its directory is added to
    ``sys.path`` during importing, and ``sys.path`` would be restored
    afterwards.

    Modules newly loaded in the same directory as *fpath* would have an
    attribute `__dynamic_loaded_by_spq__` set to 1, and fpath itself would
    have that value set to 2.

    :type fpath: str
    :param fpath: python source file path
    :type mod_name: str or None
    :param mod_name: target module name; if it exists in `sys.modules`, the
        corresponding module would be directly returned; otherwise it is added
        to ``sys.modules`` afterward. If it is None, module name would be
        derived from *fpath* by replacing '/' to '.' and special chars to '_'
    """
    fpath = os.path.realpath(fpath)

    if mod_name is None:
        # automatically generate mod_name
        mod_name = []
        for i in fpath.split(os.path.sep):
            v = ''
            for j in i:
                if not j.isidentifier() and not j.isdigit():
                    j = '_'
                v += j
            if not v.isidentifier() or keyword.iskeyword(v):
                v = '_' + v
            assert v.isidentifier() and not keyword.iskeyword(v), (
                'failed to convert to python identifier: in={} out={}'.format(
                    i, v))
            mod_name.append(v)
        mod_name = '_'.join(mod_name)

    if mod_name in sys.modules:
        return sys.modules[mod_name]

    old_path = sys.path[:]
    mod_dir = os.path.dirname(fpath)
    sys.path.append(mod_dir)
    old_mod_names = set(sys.modules.keys())

    try:
        final_mod = importlib.machinery.SourceFileLoader(
            mod_name, fpath).load_module()
    finally:
        sys.path.remove(mod_dir)

    sys.modules[mod_name] = final_mod

    for name, mod in list(sys.modules.items()):
        if name in old_mod_names:
            continue

        try:
            fpath = getattr(mod, '__file__', None)
        except Exception as exc:
            print('caught exception {} while trying to get '
                  'read __file__ attr from {}'.format(repr(exc), name))
            continue

        if fpath is not None and (
                os.path.dirname(os.path.realpath(fpath)).startswith(mod_dir)):
            try:
                mod.__dynamic_loaded_by_spq__ = 1
            except Exception:
                pass

    try:
        final_mod.__dynamic_loaded_by_spq__ = 2
    except Exception:
        pass
    return final_mod


def load_network(network, get_kwargs={}):
    '''load a model defined by model.py'''
    network = os.path.realpath(network)
    mf = magic.from_file(network, mime=True)
    mf = mf.decode('utf-8') if isinstance(mf, bytes) else mf
    if mf.startswith('text'):
        return import_python_source_as_module(network).Model().build()
    else:
        print('Only supports a model.py which defines a network')
        exit(0)

def compute_receptiveField_and_stride(nodes):
    stride_list = []
    receptive_field_list = []
    new_nodes = collections.OrderedDict()
    for k, v_dict in nodes.items():
        data_format = v_dict.get('data_format', None)
        ksize = v_dict.get('ksize', [])
        shape = v_dict.get('shape', [])
        strides = v_dict.get('strides', [])

        if data_format == 'NHWC':
            h_stride, w_stride = strides[1], strides[2]
            if ksize:
                h_size, w_size = ksize[1], ksize[2]
            else:
                h_size, w_size = shape[0], shape[1]
        elif data_format == 'NCHW':
            h_stride, w_stride = strides[2], strides[3]
            if ksize:
                h_size, w_size = ksize[2], ksize[3]
            else:
                h_size, w_size = shape[0], shape[1]
        else:
            continue

        if not stride_list:
            receptive_field_list.append((h_size, w_size))
            stride_list.append((h_stride, w_stride))
        else:
            pre_s = stride_list[-1]
            pre_rf = receptive_field_list[-1]
            stride_list.append((h_stride * pre_s[0], w_stride * pre_s[1]))
            receptive_field_list.append((h_size * pre_s[0] + pre_rf[0] - pre_s[0],
                                         w_size * pre_s[1] + pre_rf[1] - pre_s[1]))

        nodes[k].update({
            'receptive_field': receptive_field_list[-1],
            'g_stride': stride_list[-1],
        })
        new_nodes.update({k:nodes[k]})
    return new_nodes

class InfoAction:
    @classmethod
    def add_subparser(cls, subparsers):
        parser = subparsers.add_parser(
            'info', help='view some summary infomation in text')
        parser.set_defaults(func=cls.run)

    @classmethod
    def run(cls, args):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer()) # must init graph
        cls._cache = collections.OrderedDict()
        cls.param_stats(sess)
        cls.flops_stats(sess)
        cls.summary(sess)

    @classmethod
    def summary(cls, sess):
        data = [['item', 'value']]
        data.extend(list(cls._cache.items()))
        print('\n'*2)
        print('summary\n' + tabulate.tabulate(data))


    @classmethod
    def param_stats(cls, sess, bar_length_max=20):
        tot_param_dim, param_size_bit = 0, 0
        data = []
        for param in tf.trainable_variables():
            value = sess.run(param)
            param_dim = np.prod(value.shape)
            tot_param_dim += int(param_dim)
            nbits = int(re.findall(r"\d+", str(param.dtype))[0])
            param_size_bit += param_dim * nbits

            # fill data
            data.append(dict(
                name=param.name,
                shape=param.get_shape(),
                param_dim=param_dim,
                param_type=param.dtype,
                size=sizeof_fmt(param_dim * nbits / 8),
                size_cum=sizeof_fmt(tot_param_dim * nbits / 8),
                mean='{:.2g}'.format(value.mean()),
                std='{:.2g}'.format(value.std()),
            ))

        for d in data:
            ratio = d['param_dim'] / tot_param_dim
            d['ratio'] = ratio
            d['percentage'] = '{:.2f}%'.format(ratio * 100)

        # construct bar
        max_ratio = max([d['ratio'] for d in data])
        for d in data:
            bar_length = int(d['ratio'] / max_ratio * bar_length_max)
            d['size_bar'] = '#' * bar_length

        param_size = sizeof_fmt(param_size_bit / 8)
        data.append(dict(
            name='total',
            param_dim=tot_param_dim,
            size=param_size,
        ))
        cls._cache['#params'] = len(data)
        cls._cache['tot_param_dim'] = tot_param_dim
        cls._cache['param_size'] = param_size
        cls._param_size = param_size_bit / 8

        header = [
            'name', 'shape', 'mean', 'std', 'param_dim', 'size', 'size_cum',
            'percentage', 'size_bar'
        ]
        # make a table
        print('\n'*2)
        print('param stats: \n' + tabulate.tabulate(
            cls._dict2table(data, header=header)))

    @classmethod
    def _dict2table(self, list_of_dict, header):
        table_data = [header]
        for d in list_of_dict:
            row = []
            for h in header:
                v = ''
                if h in d:
                    v = d[h]
                row.append(v)
            table_data.append(row)
        return table_data

    @classmethod
    def flops_stats(cls, sess, bar_length_max=20):
        nodes = [n for n in tf.get_default_graph().as_graph_def(add_shapes=True).node]
        cls._cache['#nodes'] = len(nodes)

        # get nodes which can affect recept filed and stride
        rf_nodes = collections.OrderedDict()
        for n in nodes:
            if n.op in ['Conv2D', 'VariableV2']:
                name_scope = '/'.join(n.name.split('/')[:-1])
                if name_scope not in rf_nodes.keys():
                    rf_nodes[name_scope] = {}
                if 'shape' in n.attr.keys() and not rf_nodes[name_scope].get('shape', []):
                    rf_nodes[name_scope].update(shape=[i.size for i in n.attr['shape'].shape.dim])
                if 'strides' in n.attr.keys():
                    rf_nodes[name_scope].update(strides=list(n.attr['strides'].list.i))
                    rf_nodes[name_scope].update(data_format=n.attr['data_format'].s.decode('utf-8'))
                    rf_nodes[name_scope].update(operator=n)

            if n.op in ['MaxPool', 'AvgPool']:
                rf_nodes[n.name] = {
                    'ksize': list(n.attr['ksize'].list.i),
                    'strides': list(n.attr['ksize'].list.i),
                    'data_format': n.attr['data_format'].s.decode('utf-8'),
                    'operator': n,
                }

        rf_nodes = compute_receptiveField_and_stride(rf_nodes)

        # find the input node (only data)
        for n in nodes:
            if n.op == 'Placeholder':
                input_shape = [i.size for i in n.attr['shape'].shape.dim][1:]
                break

        for k, v_dict in rf_nodes.items():
            if v_dict['data_format'] == 'NHWC':
                v_dict['input_shape'] = input_shape
                v_dict['output_shape'] = [i.size for i in v_dict['operator'].attr['_output_shapes'].list.shape[0].dim][1:]
            elif v_dict['data_format'] == 'NCHW':
                pass

            if v_dict['operator'].op in ['Conv2D']:
                ic = v_dict['input_shape'][-1]
                v_dict['flops'] = np.prod(v_dict['output_shape']) * ic * np.prod(v_dict['shape'][:2])
            elif v_dict['operator'].op in ['MaxPool', 'AvgPool']:
                v_dict['flops'] = 0

            input_shape = v_dict['output_shape']


        opr_info = []
        total_flops = 0
        for k, v_dict in rf_nodes.items():
            total_flops += v_dict['flops']
            opr_info.append({
                'opr_name': v_dict['operator'].name,
                'opr_class': v_dict['operator'].op,
                'input_shapes': v_dict['input_shape'],
                'output_shapes': v_dict['output_shape'],
                'flops_num': v_dict['flops'],
                'flops_cum': total_flops,
                'receptive_field': v_dict['receptive_field'],
                'stride': v_dict['g_stride']
            })

        flops = [i['flops_num'] for i in opr_info]
        max_flops = max(flops + [0])
        for i in opr_info:
            f = i['flops_num']
            i['flops'] = sizeof_fmt(f, suffix='OPs')
            fc = i['flops_cum']
            i['flops_cum'] = sizeof_fmt(fc, suffix='OPs')
            r = i['ratio'] = f / total_flops
            i['percentage'] = '{:.2f}%'.format(r * 100)
            bar_length = int(f / max_flops * bar_length_max)
            i['bar'] = '#' * bar_length

        header = ['opr_name', 'opr_class', 'input_shapes', 'output_shapes', 'receptive_field',
                  'stride', 'flops', 'flops_cum', 'percentage', 'bar']

        total_flops_str = sizeof_fmt(total_flops, suffix='OPs')
        #total_var_size = sum(sum(s[1] for s in i['output_shapes']) for i in opr_info)
        opr_info.append(dict(
            opr_name='total',
            flops=total_flops_str,
            #output_shapes=total_var_size
        ))
        cls._cache['total_flops'] = total_flops_str
        cls._cache['flops/param_size'] = '{:.3g}'.format(
            total_flops / cls._param_size)

        print('\n'*2)
        print('flops stats: \n' + tabulate.tabulate(
            cls._dict2table(opr_info, header=header)))


if __name__ == "__main__":
    actions = [InfoAction,]

    parser = argparse.ArgumentParser()
    parser.add_argument('network')
    subparsers = parser.add_subparsers(help='action')

    for i in actions:
        i.add_subparser(subparsers)

    args = parser.parse_args()

    # load network
    load_network(args.network)

    if hasattr(args, 'func'):
        args.func(args)
    else:
        print('no action given')

