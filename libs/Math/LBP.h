////////////////////////////////////////////////////////////////////
// LBA.h
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef __SEACAVE_LBA_H__
#define __SEACAVE_LBA_H__


// I N C L U D E S /////////////////////////////////////////////////


// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define LBP_USE_OPENMP
#endif


namespace SEACAVE {

// S T R U C T S ///////////////////////////////////////////////////

// basic implementation of the loopy belief propagation algorithm
// based on a code originally written by Michael Waechter:
// https://github.com/nmoehrle/mvs-texturing
// Copyright(c) Michael Waechter
// Licensed under the BSD 3-Clause license
// LBP类实现，计算多标签求解问题（马尔科夫随机场）
class MATH_API LBPInference
{
public:
	typedef unsigned NodeID;  // 节点ID
	typedef unsigned LabelID; // 标签id
	typedef unsigned EdgeID;  // 边界id
	typedef float EnergyType; // 能量类型

	// data代价 
	struct DataCost {
		NodeID nodeID;   // 结点id
		EnergyType cost; // data代价
	};

	typedef EnergyType (STCALL *FncSmoothCost)(NodeID, NodeID, LabelID, LabelID);

	enum { MaxEnergy = 1000 };

protected:
	// 有向边 nodeID1->nodeID2
	struct DirectedEdge {
		NodeID nodeID1; // 起始节点
		NodeID nodeID2; // 结束节点
		std::vector<EnergyType> newMsgs; // 存放当前nodeID2为1-L label时对应的nodeID1到nodeID2的能量
		std::vector<EnergyType> oldMsgs; // 存放的上一次的nodeID2为1-L label时对应的nodeID1到nodeID2的能量
		// 初始化
		inline DirectedEdge(NodeID _nodeID1, NodeID _nodeID2) : nodeID1(_nodeID1), nodeID2(_nodeID2) {}
	};

	struct Node {
		LabelID label;      				// 节点当前的label，每次迭代时可能会被更新
		EnergyType dataCost;				// 节点当前的data代价
		std::vector<LabelID> labels;		// 节点的所有标签0,1...L
		std::vector<EnergyType> dataCosts;  // 上述标签对应的datacost
		std::vector<EdgeID> incomingEdges;  // 传向该节点的所有边
		inline Node() : label(0), dataCost(MaxEnergy) {}
	};

	std::vector<DirectedEdge> edges;  // 所有edge
	std::vector<Node> nodes;          // 所有节点
	FncSmoothCost fncSmoothCost;      // 平滑cost函数

public:
	// 初始化
	LBPInference() {}
	LBPInference(NodeID nNodes) : nodes(nNodes) {}
    // 设置节点的个数
	inline void SetNumNodes(NodeID nNodes) {
		nodes.resize(nNodes);
	}
	// 取节点的个数
	inline NodeID GetNumNodes() const {
		return (NodeID)nodes.size();
	}
	//设置邻域，添加有向边，记录每个节点的所有传向自己的边 （看课件图示）
	inline void SetNeighbors(NodeID nodeID1, NodeID nodeID2) {
		nodes[nodeID2].incomingEdges.push_back((EdgeID)edges.size());
		edges.push_back(DirectedEdge(nodeID1, nodeID2));
		nodes[nodeID1].incomingEdges.push_back((EdgeID)edges.size());
		edges.push_back(DirectedEdge(nodeID2, nodeID1));
	}
	// 设置datacost
	inline void SetDataCost(LabelID label, NodeID nodeID, EnergyType cost) {
		Node& node = nodes[nodeID];
		node.labels.push_back(label);
		const EnergyType dataCost(cost);
		node.dataCosts.push_back(dataCost);
		if (dataCost < node.dataCost) {
			node.label = label;
			node.dataCost = dataCost;
		}
		// 初始化每个进入的边的cost为0
		for (EdgeID edgeID: node.incomingEdges) {
			DirectedEdge& incomingEdge = edges[edgeID];
			incomingEdge.oldMsgs.push_back(0);
			incomingEdge.newMsgs.push_back(0);
		}
	}
	inline void SetDataCost(LabelID label, const DataCost& cost) {
		SetDataCost(label, cost.nodeID, cost.cost);
	}
	inline void SetDataCosts(LabelID label, const std::vector<DataCost>& costs) {
		for (const DataCost& cost: costs)
			SetDataCost(label, cost);
	}

	inline void SetSmoothCost(FncSmoothCost func) {
		fncSmoothCost = func;
	}
	// 能量计算：datacost+edge_smooth_cost
	EnergyType ComputeEnergy() const {
		EnergyType energy(0);
		#ifdef LBP_USE_OPENMP
		#pragma omp parallel for reduction(+:energy)
		#endif
		// 每个节点当前label的cost加和
		for (int_t nodeID = 0; nodeID < (int_t)nodes.size(); ++nodeID)
			energy += nodes[nodeID].dataCost;
		#ifdef LBP_USE_OPENMP
		#pragma omp parallel for reduction(+:energy)
		#endif
		// 继续加和当前label下每个边的cost
		for (int_t edgeID = 0; edgeID < (int_t)edges.size(); ++edgeID) {
			const DirectedEdge& edge = edges[edgeID];
			energy += fncSmoothCost(edge.nodeID1, edge.nodeID2, nodes[edge.nodeID1].label, nodes[edge.nodeID2].label);
		}
		return energy;
	}
	/**
	 * @brief lbp 迭代优化关键函数，
	 * 
	 * @param[in] num_iterations 迭代次数
	 */
	void Optimize(unsigned num_iterations) {
		for (unsigned i = 0; i < num_iterations; ++i) {
			#ifdef LBP_USE_OPENMP
			#pragma omp parallel for
			#endif
			// Step 1 计算每个边的end的节点的每个label的最小能量值
			for (int_t edgeID = 0; edgeID < (int_t)edges.size(); ++edgeID) {
				DirectedEdge& edge = edges[edgeID];
				// 边的方向是id1->id2
				const std::vector<LabelID>& labels1 = nodes[edge.nodeID1].labels;
				const std::vector<LabelID>& labels2 = nodes[edge.nodeID2].labels;
				for (size_t j = 0; j < labels2.size(); ++j) {
					const LabelID label2(labels2[j]);
					EnergyType minEnergy(std::numeric_limits<EnergyType>::max());
					// 计算当前节点nodeID2的label为j时，nodeID1为0-L中的哪个label时cost最小
					for (size_t k = 0; k < labels1.size(); ++k) {
						const LabelID label1(labels1[k]);
						EnergyType energy(nodes[edge.nodeID1].dataCosts[k] + fncSmoothCost(edge.nodeID1, edge.nodeID2, label1, label2));
						const std::vector<EdgeID>& incoming_edges1 = nodes[edge.nodeID1].incomingEdges;
						// 计算nodeID1的各个进入边的cost之和
						for (size_t n = 0; n < incoming_edges1.size(); ++n) {
							const DirectedEdge& pre_edge = edges[incoming_edges1[n]];
							if (pre_edge.nodeID1 == edge.nodeID2) continue;
							energy += pre_edge.oldMsgs[k];
						}
						// 找能量最小的值
						if (minEnergy > energy)
							minEnergy = energy;
					}
					// 得到当前节点nodeID2的label为j时，进入边id1->id2传递的最小能量
					edge.newMsgs[j] = minEnergy;
				}
			}
			#ifdef LBP_USE_OPENMP
			#pragma omp parallel for
			#endif
			// newMsgs的值传给oldMsgs，并减去最小值：做归一化，避免避免能量值太大溢出
			for (int_t edgeID = 0; edgeID < (int_t)edges.size(); ++edgeID) {
				DirectedEdge& edge = edges[edgeID];
				edge.newMsgs.swap(edge.oldMsgs);
				EnergyType minMsg(std::numeric_limits<EnergyType>::max());
				for (EnergyType msg: edge.oldMsgs)
					if (minMsg > msg)
						minMsg = msg;
				for (EnergyType& msg: edge.oldMsgs)
					msg -= minMsg;
			}
		}
		#ifdef LBP_USE_OPENMP
		#pragma omp parallel for
		#endif
		// Step 2 遍历每个节点，判断节点0-L个label值的最小能量值，把最小的对应的label和cost赋值给当前节点
		for (int_t nodeID = 0; nodeID < (int_t)nodes.size(); ++nodeID) {
			Node& node = nodes[nodeID];
			EnergyType minEnergy(std::numeric_limits<EnergyType>::max());
			for (size_t j = 0; j < node.labels.size(); ++j) {
				EnergyType energy(node.dataCosts[j]);
				// 将所有进入该节点的边的能量值加和即为当前节点的当前label的能量值
				for (EdgeID incoming_edge_idx : node.incomingEdges)
					energy += edges[incoming_edge_idx].oldMsgs[j];
				// 取最小能量对应的label
				if (energy < minEnergy) {
					minEnergy = energy;
					node.label = node.labels[j];
					node.dataCost = node.dataCosts[j];
				}
			}
		}
	}

	EnergyType Optimize() {
		TD_TIMER_STARTD();
		EnergyType energy(ComputeEnergy());
		EnergyType diff(energy);
		unsigned i(0);
		#if 1
		unsigned nIncreases(0), nTotalIncreases(0);
		#endif
		while (true) {
			TD_TIMER_STARTD();
			const EnergyType last_energy(energy);
			// 计算新的代价值
			Optimize(1);
			// 计算当前label分配下，所有节点的代价之和
			energy = ComputeEnergy();
			diff = last_energy - energy;
			DEBUG_ULTIMATE("\t%2u. e: %g\td: %g\tt: %s", i, last_energy, diff, TD_TIMER_GET_FMT().c_str());
			// 迭代次数超过100 或上次和当前能量没有变化则优化结束退出循环
			if (++i > 100 || diff == EnergyType(0))
				break;
			#if 1
			// 如果能量反向增加了 nIncreases超过3就退出，反之nTotalIncreases>5也退出
			if (diff < EnergyType(0)) {
				++nTotalIncreases;
				if (++nIncreases > 3)
					break;
			} else {
				if (nTotalIncreases > 5)
					break;
				nIncreases = 0;
			}
			#else
			if (diff < EnergyType(0))
				break;
			#endif
		}
		if (diff == EnergyType(0)) {
			DEBUG_ULTIMATE("Inference converged in %u iterations: %g energy (%s)", i, energy, TD_TIMER_GET_FMT().c_str());
		} else {
			DEBUG_ULTIMATE("Inference aborted (energy increased): %u iterations, %g energy (%s)", i, energy, TD_TIMER_GET_FMT().c_str());
		}
		return energy;
	}

	inline LabelID GetLabel(NodeID nodeID) const {
		return nodes[nodeID].label;
	}
};
/*----------------------------------------------------------------*/

} // namespace SEACAVE

#endif // __SEACAVE_LBP_H__
