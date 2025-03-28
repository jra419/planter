########################################################################
# THIS FILE IS PART OF Planter PROJECT
# Copyright (c) Changgang Zheng and Computing Infrastructure Group
# Department of Engineering Science, University of Oxford
# All rights reserved.
# E-mail: changgang.zheng@eng.ox.ac.uk or changgangzheng@qq.com
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at :
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#########################################################################
# This file was autogenerated

/*
 * Planter
 *
 * This program implements a simple protocol. It can be carried over Ethernet
 * (Ethertype 0x1234).
 *
 * The Protocol header looks like this:
 *
 *        0                1                  2              3
 * +----------------+----------------+----------------+---------------+
 * |      P         |       4        |     Version    |     Type      |
 * +----------------+----------------+----------------+---------------+
 * |                              feature0                            |
 * +----------------+----------------+----------------+---------------+
 * |                              feature1                            |
 * +----------------+----------------+----------------+---------------+
 * |                              feature2                            |
 * +----------------+----------------+----------------+---------------+
 * |                              feature3                            |
 * +----------------+----------------+----------------+---------------+
 * |                              Result                              |
 * +----------------+----------------+----------------+---------------+
 *
 * P is an ASCII Letter 'P' (0x50)
 * 4 is an ASCII Letter '4' (0x34)
 * Version is currently 1 (0x01)
 * Type is currently 1 (0x01)
 *
 * The device receives a packet, do the classification, fills in the
 * result and sends the packet back out of the same port it came in on, while
 * swapping the source and destination addresses.
 *
 * If an unknown operation is specified or the header is not valid, the packet
 * is dropped
 */

#define CLASS_NOT_SET 10

#include <core.p4>
#include <tna.p4>

/*************************************************************************
*********************** headers and metadata******************************
*************************************************************************/

const bit<16> ETHERTYPE_Planter = 0x1234;
const bit<8>  Planter_P     = 0x50;   // 'P'
const bit<8>  Planter_4     = 0x34;   // '4'
const bit<8>  Planter_VER   = 0x01;   // v0.1

header ethernet_h {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}

header Planter_h{
    bit<8> p;
    bit<8> four;
    bit<8> ver;
    bit<8> typ;
    bit<32> feature0;
    bit<32> feature1;
    bit<32> feature2;
    bit<32> feature3;
    bit<32> result;
}

struct header_t {
    ethernet_h   ethernet;
    Planter_h    Planter;
}

struct metadata_t {
#define CLASS_NOT_SET 10

    bit<16> tree_1_vote;
    bit<16> node_id;
    bit<16> prevFeature;
    bit<16> isTrue;
    bit<32>  DstAddr;
    bit<32> feature0;
    bit<32> feature1;
    bit<32> feature2;
    bit<32> feature3;
    bit<32> result;
    bit<8> flag ;
}

/*************************************************************************
*********************** Ingress Parser ***********************************
*************************************************************************/

parser SwitchIngressParser(
    packet_in pkt,
    out header_t hdr,
    out metadata_t meta,
    out ingress_intrinsic_metadata_t ig_intr_md) {

    state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
        ETHERTYPE_Planter : check_planter_version;
        default           : accept;
        }
    }

    state check_planter_version {
        transition select(pkt.lookahead<Planter_h>().p,
                          pkt.lookahead<Planter_h>().four,
                          pkt.lookahead<Planter_h>().ver) {
        (Planter_P, Planter_4, Planter_VER) : parse_planter;
        default                             : accept;
        }
    }

    state parse_planter {
        pkt.extract(hdr.Planter);
        meta.feature0 = hdr.Planter.feature0;
        meta.feature1 = hdr.Planter.feature1;
        meta.feature2 = hdr.Planter.feature2;
        meta.feature3 = hdr.Planter.feature3;
        meta.flag = 1 ;
        transition accept;
    }
}

/*************************************************************************
*********************** Ingress Deparser *********************************
**************************************************************************/

control SwitchIngressDeparser(
    packet_out pkt,
    inout header_t hdr,
    in metadata_t ig_md,
    in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    apply {
        pkt.emit(hdr);
    }
}

/*************************************************************************
*********************** Egress Parser ***********************************
*************************************************************************/

parser SwitchEgressParser(
    packet_in pkt,
    out header_t hdr,
    out metadata_t meta,
    out egress_intrinsic_metadata_t eg_intr_md) {
    state start {
        pkt.extract(eg_intr_md);
        transition accept;
        }

}

/*************************************************************************
*********************** Egress Deparser *********************************
**************************************************************************/

control SwitchEgressDeparser(
    packet_out pkt,
    inout header_t hdr,
    in metadata_t eg_md,
    in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md) {
    apply {
        pkt.emit(hdr);
    }
}

/*************************************************************************
*********************** Ingress Processing********************************
**************************************************************************/

control SwitchIngress(
    inout header_t hdr,
    inout metadata_t meta,
    in ingress_intrinsic_metadata_t ig_intr_md,
    in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    action drop() {
        ig_dprsr_md.drop_ctl = 0x1;
    }

    action send(PortId_t port) {
        ig_tm_md.ucast_egress_port = port;
    }

    action CheckFeature(bit<16> node_id, bit<16> f_inout, bit<32> threshold) {
        bit<32> feature = 0;
        bit<16> f = f_inout ;
        if (f == 0) {
            feature = hdr.Planter.feature0;
        }
        if (f == 1) {
            feature = hdr.Planter.feature1;
        }
        if (f == 2) {
            feature = hdr.Planter.feature2;
        }
        if (f == 3) {
            feature = hdr.Planter.feature3;
        }
        bit<32> th = threshold - feature;
        if (th & 0b10000000000000000000000000000000==0){
            meta.isTrue = 1;
        }else{
            meta.isTrue = 0;
        }
        meta.prevFeature = f;
        meta.node_id = node_id;
    }

    action SetClass1(bit <16> node_id, bit <16> class ) {
        meta.tree_1_vote = class;
        meta.node_id = node_id; // just for debugging otherwise not needed
    }
    table level_1_1{
        key = {
            meta.node_id: exact;
            meta.prevFeature: exact;
            meta.isTrue: exact;
        }
        actions = {
            NoAction;
            CheckFeature;
            SetClass1;
        }
        size = 1024;
    }

    table level_1_2{
        key = {
            meta.node_id: exact;
            meta.prevFeature: exact;
            meta.isTrue: exact;
        }
        actions = {
            NoAction;
            CheckFeature;
            SetClass1;
        }
        size = 1024;
    }

    table level_1_3{
        key = {
            meta.node_id: exact;
            meta.prevFeature: exact;
            meta.isTrue: exact;
        }
        actions = {
            NoAction;
            CheckFeature;
            SetClass1;
        }
        size = 1024;
    }

    table level_1_4{
        key = {
            meta.node_id: exact;
            meta.prevFeature: exact;
            meta.isTrue: exact;
        }
        actions = {
            NoAction;
            CheckFeature;
            SetClass1;
        }
        size = 1024;
    }

    table level_1_5{
        key = {
            meta.node_id: exact;
            meta.prevFeature: exact;
            meta.isTrue: exact;
        }
        actions = {
            NoAction;
            CheckFeature;
            SetClass1;
        }
        size = 1024;
    }

    action read_lable(bit<32> label){
        hdr.Planter.result = label;
    }

    action write_default_decision() {
        hdr.Planter.result = 0;
    }

    table decision {
        key = { meta.tree_1_vote:exact;
                }
        actions={
            read_lable;
            write_default_decision;
        }
        size = 2;
        default_action = write_default_decision;
    }

    apply{
        meta.tree_1_vote = CLASS_NOT_SET;

        meta.node_id = 0;
        meta.prevFeature = 0;
        meta.isTrue = 1;
        level_1_1.apply();
        if (meta.tree_1_vote == CLASS_NOT_SET) {
          level_1_2.apply();
          if (meta.tree_1_vote == CLASS_NOT_SET) {
            level_1_3.apply();
            if (meta.tree_1_vote == CLASS_NOT_SET) {
              level_1_4.apply();
              if (meta.tree_1_vote == CLASS_NOT_SET) {
                level_1_5.apply();
        } } } } 

        decision.apply();
        send(ig_intr_md.ingress_port);
    }
}
/*************************************************************************
*********************** egress Processing********************************
**************************************************************************/

control SwitchEgress(inout header_t hdr,
    inout metadata_t meta,
    in egress_intrinsic_metadata_t eg_intr_md,
    in egress_intrinsic_metadata_from_parser_t eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t     eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t  eg_oport_md) {

    action drop() {
        eg_dprsr_md.drop_ctl = 0x1;
    }

    apply {
    }
}
/*************************************************************************
***********************  S W I T C H  ************************************
*************************************************************************/

Pipeline(SwitchIngressParser(),
    SwitchIngress(),
    SwitchIngressDeparser(),
    SwitchEgressParser(),
    SwitchEgress(),
    SwitchEgressDeparser()) pipe;

Switch(pipe) main;