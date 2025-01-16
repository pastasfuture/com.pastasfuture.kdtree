using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;

namespace Pastasfuture.KDTree.Runtime
{
    public static class KDTree
    {
        public struct KDTreeData
        {
            public NativeArray<int> payloadIndices;
            public NativeArray<float3> positions;
            public NativeArray<float> radii;
            public NativeArray<float3> aabbs;
            public NativeArray<float> splitPositions;
            public NativeArray<byte> splitAxes;
            public NativeArray<int2> nodeRanges;

            public static readonly KDTreeData zero = new KDTreeData();
        }

        public struct KDTreeHeader
        {
            public int count;
            public int holeCount;
            public int depthCount;
            public int depthCapacity;

            public static readonly KDTreeHeader zero = new KDTreeHeader()
            {
                count = 0,
                holeCount = 0,
                depthCount = 0,
                depthCapacity = 0
            };
        }

        public static void Dispose(ref KDTreeHeader header, ref KDTreeData data)
        {
            if (data.payloadIndices.IsCreated) { data.payloadIndices.Dispose(); }
            if (data.positions.IsCreated) { data.positions.Dispose(); }
            if (data.radii.IsCreated) { data.radii.Dispose(); }
            if (data.aabbs.IsCreated) { data.aabbs.Dispose(); }
            if (data.splitPositions.IsCreated) { data.splitPositions.Dispose(); }
            if (data.splitAxes.IsCreated) { data.splitAxes.Dispose(); }
            if (data.nodeRanges.IsCreated) { data.nodeRanges.Dispose(); }

            header.count = 0;
            header.holeCount = 0;
            header.depthCount = 0;
            header.depthCapacity = 0;
        }
        
        public static JobHandle Dispose(ref KDTreeData data, JobHandle jobHandle)
        {
            JobHandle disposalJobs = default;
            if (data.payloadIndices.IsCreated) { disposalJobs = data.payloadIndices.Dispose(jobHandle); }
            if (data.positions.IsCreated) { disposalJobs = JobHandle.CombineDependencies(disposalJobs, data.positions.Dispose(jobHandle)); }
            if (data.radii.IsCreated) { disposalJobs = JobHandle.CombineDependencies(disposalJobs, data.radii.Dispose(jobHandle)); }
            if (data.aabbs.IsCreated) { disposalJobs = JobHandle.CombineDependencies(disposalJobs, data.aabbs.Dispose(jobHandle)); }
            if (data.splitPositions.IsCreated) { disposalJobs = JobHandle.CombineDependencies(disposalJobs, data.splitPositions.Dispose(jobHandle)); }
            if (data.splitAxes.IsCreated) { disposalJobs = JobHandle.CombineDependencies(disposalJobs, data.splitAxes.Dispose(jobHandle)); }
            if (data.nodeRanges.IsCreated) { disposalJobs = JobHandle.CombineDependencies(disposalJobs, data.nodeRanges.Dispose(jobHandle)); }
            return disposalJobs;
        }

        [BurstCompile]
        public static int Count(ref KDTreeHeader header, ref KDTreeData data)
        {
            return IsCreated(ref header, ref data) ? header.count : 0;
        }

        [BurstCompile]
        public static int CountValid(ref KDTreeHeader header, ref KDTreeData data)
        {
            return IsCreated(ref header, ref data) ? (header.count - header.holeCount) : 0;
        }

        [BurstCompile]
        public static int Capacity(ref KDTreeHeader header, ref KDTreeData data)
        {
            return IsCreated(ref header, ref data) ? data.positions.Length : 0;
        }

        [BurstCompile]
        public static bool IsCreated(ref KDTreeHeader header, ref KDTreeData data)
        {
            return data.positions.IsCreated;
        }

        public static void Allocate(ref KDTreeHeader header, ref KDTreeData data, int capacity, int depthCapacity, Allocator allocator)
        {
            Debug.Assert(!IsCreated(ref header, ref data));

            data.payloadIndices = new NativeArray<int>(capacity, allocator);
            data.positions = new NativeArray<float3>(capacity, allocator);
            data.radii = new NativeArray<float>(capacity, allocator);
            header.count = 0;
            header.holeCount = 0;

            int boundsCount = 1 << depthCapacity;
            data.aabbs = new NativeArray<float3>(boundsCount * 2, allocator);
            data.splitPositions = new NativeArray<float>(boundsCount >> 1, allocator);
            data.splitAxes = new NativeArray<byte>(boundsCount >> 1, allocator);
            data.nodeRanges = new NativeArray<int2>(boundsCount, allocator);
            header.depthCapacity = depthCapacity;
            header.depthCount = 0;
        }

        public static void Copy(ref KDTreeHeader headerSrc, ref KDTreeData dataSrc, ref KDTreeHeader headerDst, ref KDTreeData dataDst, int count, int depthCount)
        {
            Debug.Assert(Capacity(ref headerSrc, ref dataSrc) >= count);
            Debug.Assert(Capacity(ref headerDst, ref dataDst) >= count);
            Debug.Assert(headerSrc.count >= count);

            NativeArray<int>.Copy(dataSrc.payloadIndices, dataDst.payloadIndices, count);
            NativeArray<float3>.Copy(dataSrc.positions, dataDst.positions, count);
            NativeArray<float>.Copy(dataSrc.radii, dataDst.radii, count);

            int boundsCount = 1 << depthCount;
            Debug.Assert(depthCount <= headerSrc.depthCapacity);
            Debug.Assert(depthCount <= headerDst.depthCapacity);
            NativeArray<float3>.Copy(dataSrc.aabbs, dataDst.aabbs, boundsCount);
            NativeArray<float>.Copy(dataSrc.splitPositions, dataDst.splitPositions, boundsCount >> 1);
            NativeArray<byte>.Copy(dataSrc.splitAxes, dataDst.splitAxes, boundsCount >> 1);
            NativeArray<int2>.Copy(dataSrc.nodeRanges, dataDst.nodeRanges, boundsCount);

            headerDst.count = count;
            headerDst.holeCount = headerSrc.holeCount;
            headerDst.depthCount = depthCount;
        }

        public static void EnsureCapacity(ref KDTreeHeader header, ref KDTreeData data, int capacityRequested, int depthCapacityRequested, Allocator allocator)
        {
            if (Capacity(ref header, ref data) >= capacityRequested
                && header.depthCapacity >= depthCapacityRequested) { return; }

            KDTreeData dataNext = new KDTreeData();
            KDTreeHeader headerNext = new KDTreeHeader();
            Allocate(ref headerNext, ref dataNext, capacityRequested, depthCapacityRequested, allocator);
            if (IsCreated(ref header, ref data))
            {
                Copy(ref header, ref data, ref headerNext, ref dataNext, header.count, math.min(depthCapacityRequested, header.depthCount));
                Dispose(ref header, ref data);
            }
            data = dataNext;
            header = headerNext;
        }

        public static void Add(ref KDTreeHeader header, ref KDTreeData data, int index, float3 position, float radius, Allocator allocator)
        {
            EnsureCapacity(ref header, ref data, header.count + 1, header.depthCapacity, allocator);

            data.payloadIndices[header.count] = index;
            data.positions[header.count] = position;
            data.radii[header.count] = radius;

            ++header.count;
        }

        [BurstCompile]
        public static bool TryAdd(ref KDTreeHeader header, ref KDTreeData data, int index, float3 position, float radius)
        {
            if (Capacity(ref header, ref data) < (header.count + 1)) { return false; }

            data.payloadIndices[header.count] = index;
            data.positions[header.count] = position;
            data.radii[header.count] = radius;

            ++header.count;

            return true;
        }

        [BurstCompile]
        public static bool TryAddHoleForPadding(ref KDTreeHeader header, ref KDTreeData data, int index)
        {
            if (Capacity(ref header, ref data) < (header.count + 1)) { return false; }

            data.payloadIndices[header.count] = index;
            data.positions[header.count] = float3.zero;
            data.radii[header.count] = 0.0f;
            
            ++header.holeCount;
            data.payloadIndices[header.count] = -1;

            ++header.count;

            return true;
        }

        [BurstCompile]
        public static bool IndexIsValid(ref KDTreeHeader header, ref KDTreeData data, int index)
        {
            return data.payloadIndices[index] != -1;
        }

        [BurstCompile]
        public static void RemoveAtPayloadIndex(ref KDTreeHeader header, ref KDTreeData data, int payloadIndex)
        {
            Debug.Assert(IsCreated(ref header, ref data));

            int index = FindIndexFromPayloadIndex(ref header, ref data, payloadIndex);
            Debug.Assert(index != -1);

            // Simply set our payload index to -1 to make KDTree treat this slot as a hole,
            // which will be skipped over when operating with the KDTree.
            // It is up to the user to query holeCount and decide when to rebuild the tree to improve query efficiency.
            ++header.holeCount;
            data.payloadIndices[index] = -1;
        }

        [BurstCompile]
        public static int FindIndexFromPayloadIndex(ref KDTreeHeader header, ref KDTreeData data, int payloadIndex)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(payloadIndex >= 0);

            for (int i = 0, iLen = header.count; i < iLen; ++i)
            {
                int payloadIndexCurrent = data.payloadIndices[i];
                if (payloadIndexCurrent == payloadIndex)
                {
                    return i;
                }
            }

            return -1;
        }

        [BurstCompile]
        public static void Clear(ref KDTreeHeader header, ref KDTreeData data)
        {
            header.count = 0;
            header.holeCount = 0;
        }

        [BurstCompile]
        public static bool IsLeaf(ref KDTreeHeader header, ref KDTreeData data, int depth)
        {
            Debug.Assert(IsCreated(ref header, ref data));

            return (depth + 1) == header.depthCount;
        }

        [BurstCompile]
        public static float ComputeSortKey(ref KDTreeHeader header, ref KDTreeData data, int i, int sortAxis)
        {
            Debug.Assert(i >= 0);
            Debug.Assert(i < header.count);
            Debug.Assert(sortAxis >= 0 && sortAxis <= 2);

            return IndexIsValid(ref header, ref data, i)
                ? data.positions[i][sortAxis]
                : float.MaxValue; // throw holes to the end of array.
        }

        [BurstCompile]
        public static void Swap(ref KDTreeHeader header, ref KDTreeData data, int indexA, int indexB)
        {
            Debug.Assert(indexA < Capacity(ref header, ref data));
            Debug.Assert(indexB < Capacity(ref header, ref data));

            (data.payloadIndices[indexA], data.payloadIndices[indexB]) = (data.payloadIndices[indexB], data.payloadIndices[indexA]);
            (data.positions[indexA], data.positions[indexB]) = (data.positions[indexB], data.positions[indexA]);
            (data.radii[indexA], data.radii[indexB]) = (data.radii[indexB], data.radii[indexA]);
        }

        
        [BurstCompile]
        private static void Quickselect(ref KDTreeHeader header, ref KDTreeData data, int left, int right, int n, int sortAxis)
        {
            while (left < right)
            {
                int pivotIndex = ((right - left) >> 1) + left;
                float pivotKey = ComputeSortKey(ref header, ref data, pivotIndex, sortAxis);

                // Move the pivot to the end
                Swap(ref header, ref data, pivotIndex, right);

                // Three-way partitioning (to improve performance with many identical keys, which occurs when there are many empty slots in the tree.
                int less = left;
                int equal = left;
                int greater = right;

                while (equal <= greater)
                {
                    float keyCurrent = ComputeSortKey(ref header, ref data, equal, sortAxis);

                    if (keyCurrent < pivotKey)
                    {
                        Swap(ref header, ref data, less, equal);
                        less++;
                        equal++;
                    }
                    else if (keyCurrent > pivotKey)
                    {
                        Swap(ref header, ref data, equal, greater);
                        greater--;
                    }
                    else
                    {
                        equal++;
                    }
                }

                if (n < less)
                {
                    right = less - 1;
                }
                else if (n > greater)
                {
                    left = greater + 1;
                }
                else
                {
                    break;
                }
            }
        }

        [BurstCompile]
        public static void ComputeAABBFromRange(out float3 aabbMin, out float3 aabbMax, ref KDTreeHeader header, ref KDTreeData data, int left, int right)
        {
            aabbMin = float.MaxValue;
            aabbMax = -float.MaxValue;

            Debug.Assert(left >= 0 && left < header.count);
            Debug.Assert(right >= 0 && right < header.count);

            for (int i = left; i <= right; ++i)
            {
                if (!IndexIsValid(ref header, ref data, i)) { continue; } // Item has been removed. Skip.
                if (float.IsNaN(data.positions[i].x) || float.IsNaN(data.positions[i].y) || float.IsNaN(data.positions[i].z)) { continue; }
                if (float.IsNaN(data.radii[i])) { continue; }

                aabbMin = math.min(aabbMin, data.positions[i] - data.radii[i]);
                aabbMax = math.max(aabbMax, data.positions[i] + data.radii[i]);
            }
        }

        [BurstCompile]
        private static int ComputeSplitAxisFromAABB(float3 aabbMin, float3 aabbMax)
        {
            float3 extents = aabbMax - aabbMin;

            return (extents[0] > extents[1])
                ? ((extents[0] > extents[2]) ? 0 : 2)
                : ((extents[1] > extents[2]) ? 1 : 2);
        }

        public enum KDTreePartitionScheme : uint
        {
            AxisMax = 0,
            SeparationMax,
            SurfaceAreaHeuristic
        }

        [BurstCompile]
        public static void Build(ref KDTreeHeader header, ref KDTreeData data, int depthCount, KDTreePartitionScheme partitionScheme = KDTreePartitionScheme.AxisMax)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);
            Debug.Assert(header.depthCapacity >= depthCount);
            Debug.Assert(depthCount > 0);

            // data.count can reach zero after CompactHoles is executed.
            if (header.count == 0) { return; }

            header.depthCount = math.clamp(depthCount, 1, CeilLog2(header.count));

            BuildRecurse(ref header, ref data, depth: 0, nodeIndex: 0, left: 0, right: header.count - 1, partitionScheme);      
        }

        [BurstCompile]
        private static void BuildRecurse(ref KDTreeHeader header, ref KDTreeData data, int depth, int nodeIndex, int left, int right, KDTreePartitionScheme partitionScheme)
        {
            data.nodeRanges[nodeIndex] = new int2(left, right);
            
            Debug.Assert(left >= 0 && left < header.count);
            Debug.Assert(right >= 0 && right < header.count);
            Debug.Assert(right >= left);

            ComputeAABBFromRange(out float3 aabbMin, out float3 aabbMax, ref header, ref data, left, right);
            data.aabbs[nodeIndex * 2 + 0] = aabbMin;
            data.aabbs[nodeIndex * 2 + 1] = aabbMax;
            
            // Only perform split if we are not leaf node.
            if ((depth + 1) < header.depthCount)
            {
                Debug.Assert((left + 1) < right);
                int sortAxis = -1;
                int pivot = -1;
                switch (partitionScheme)
                {
                    case KDTreePartitionScheme.AxisMax:
                    {
                        pivot = PartitionLargestAxis(out sortAxis, ref header, ref data, left, right, aabbMin, aabbMax);
                        break;
                    }

                    case KDTreePartitionScheme.SeparationMax:
                    {
                        pivot = PartitionSeparationMax(out sortAxis, ref header, ref data, left, right);
                        break;
                    }

                    case KDTreePartitionScheme.SurfaceAreaHeuristic:
                    {
                        pivot = PartitionSAH(out sortAxis, ref header, ref data, left, right, aabbMin, aabbMax);
                        break;
                    }

                    default:
                    {
                        Debug.Assert(false);
                        break;
                    }
                }

                Debug.Assert(left <= pivot);
                Debug.Assert(pivot <= right);

                data.splitAxes[nodeIndex] = (byte)sortAxis;
                data.splitPositions[nodeIndex] = data.positions[pivot][sortAxis];
                
                // Recurse into nearest child.
                // (1 << 0) == 1
                // (1 << 1) == 2
                // (1 << 2) == 4
                // (1 << 4) == 8
                //
                //          0
                //    1            2
                //  3   4       5     6
                // 7 8 9 10   11 12 13 14
                int nodeIndexChildLeft = (int)((nodeIndex + 1) << 1) - 1;
                int nodeIndexChildRight = nodeIndexChildLeft + 1;

                // Switch to the fastest partitioning scheme deeper in the tree.
                KDTreePartitionScheme partitionSchemeNext = (depth > 8) ? KDTreePartitionScheme.AxisMax : partitionScheme;
                
                BuildRecurse(ref header, ref data, depth: depth + 1, nodeIndex: nodeIndexChildLeft, left: left, right: pivot - 1, partitionSchemeNext);
                BuildRecurse(ref header, ref data, depth: depth + 1, nodeIndex: nodeIndexChildRight, left: pivot, right: right, partitionSchemeNext);
            }            
        }
        
        [BurstCompile]
        private static int PartitionLargestAxis(out int sortAxis, ref KDTreeHeader header, ref KDTreeData data, int left, int right, float3 aabbMin, float3 aabbMax)
        {
            sortAxis = ComputeSplitAxisFromAABB(aabbMin, aabbMax);
            int pivot = ((right - left) >> 1) + left;
            Quickselect(ref header, ref data, left, right, pivot, sortAxis);
            return pivot;
        }

        [BurstCompile]
        private static int PartitionSeparationMax(out int sortAxis, ref KDTreeHeader header, ref KDTreeData data, int left, int right)
        {
            float separationMax = float.MinValue;
            sortAxis = -1;
            int pivot = ((right - left) >> 1) + left;

            for (int axis = 0; axis < 3; axis++)
            {
                Quickselect(ref header, ref data, left, right, pivot, axis);

                ComputeAABBFromRange(out float3 leftMin, out float3 leftMax, ref header, ref data, left, pivot - 1);
                ComputeAABBFromRange(out float3 rightMin, out float3 rightMax, ref header, ref data, pivot, right);

                float separationCurrent = rightMin[axis] - leftMax[axis];

                if (separationCurrent > separationMax)
                {
                    separationMax = separationCurrent;
                    sortAxis = axis;
                }
            }

            Debug.Assert(sortAxis != -1);
            Quickselect(ref header, ref data, left, right, pivot, sortAxis);
            return pivot;
        }

        [BurstCompile]
        private static int PartitionSAH(out int sortAxis, ref KDTreeHeader header, ref KDTreeData data, int left, int right, float3 aabbMin, float3 aabbMax)
        {
            const float traversalCost = 2.0f;
            const float intersectionCost = 1.0f;

            float sahBest = float.MaxValue;
            int pivotBest = -1;
            sortAxis = -1;

            float surfaceAreaParent = ComputeAABBSurfaceArea(aabbMin, aabbMax);
            float surfaceAreaParentInverse = (surfaceAreaParent > 1e-5f) ? (1.0f / surfaceAreaParent) : 0.0f;

            for (int axis = 0; axis < 3; axis++)
            {
                int pivot = ((right - left) >> 1) + left;
                Quickselect(ref header, ref data, left, right, pivot, axis);
                ComputeAABBFromRange(out float3 leftMin, out float3 leftMax, ref header, ref data, left, pivot - 1);
                ComputeAABBFromRange(out float3 rightMin, out float3 rightMax, ref header, ref data, pivot, right);

                float leftArea = ComputeAABBSurfaceArea(leftMin, leftMax);
                float rightArea = ComputeAABBSurfaceArea(rightMin, rightMax);

                float sah = traversalCost + intersectionCost * surfaceAreaParentInverse * ((leftArea * (pivot - left)) + (rightArea * ((right - pivot) + 1)));

                if (sah < sahBest)
                {
                    sahBest = sah;
                    pivotBest = pivot;
                    sortAxis = axis;
                }
            }

            Quickselect(ref header, ref data, left, right, pivotBest, sortAxis);
            return pivotBest;
        }

        [BurstCompile]
        private static float ComputeAABBSurfaceArea(float3 min, float3 max)
        {
            float3 size = max - min;
            return size.x * size.z * 2.0f + size.x * size.y * 2.0f + size.y * size.z * 2.0f;
        }

        [BurstCompile]
        public static void CompactHoles(ref KDTreeHeader header, ref KDTreeData data)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);

            if (header.holeCount == 0) { return; }

            int left = 0;
            int right = header.count - 1;
            int pivot = right - header.holeCount;
            int sortAxis = 0; // Sort axis can be any axis, we only care about partitioning holes to the end.
            Quickselect(ref header, ref data, left, right, pivot, sortAxis);

            // Debug:
            for (int i = pivot + 1; i <= right; ++i)
            {
                Debug.Assert(!IndexIsValid(ref header, ref data, i));
            }

            // All holes are now partitioned to the end of the array.
            // We can shink counts and clear out holes.
            header.count -= header.holeCount;
            header.holeCount = 0;
        }

        [BurstCompile]
        public static void FindNearestNeighbor(out int index, out float distanceSquared, ref KDTreeHeader header, ref KDTreeData data, float3 position, float distanceSquaredMax = float.MaxValue)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);
            Debug.Assert(header.depthCount > 0);

            index = -1;
            distanceSquared = distanceSquaredMax;
            KDTreeTraversalStateGeneralizedOptimized state = KDTreeTraversalStateGeneralizedOptimized.zero;
            while (FindNearestNeighborStacklessGeneralizedIteratorOptimized(out int left, out int right, distanceSquared, ref state, ref header, ref data, position))
            {
                for (int i = left; i <= right; ++i)
                {
                    if (!IndexIsValid(ref header, ref data, i)) { continue; } // Item has been removed. Skip.
                    if (float.IsNaN(data.positions[i].x) || float.IsNaN(data.positions[i].y) || float.IsNaN(data.positions[i].z)) { continue; }
                    if (float.IsNaN(data.radii[i])) { continue; }

                    float distanceCurrent = math.max(0.0f, math.length(data.positions[i] - position) - data.radii[i]);
                    float distanceSquaredCurrent = distanceCurrent * distanceCurrent;
                    if (distanceSquaredCurrent < distanceSquared)
                    {
                        distanceSquared = distanceSquaredCurrent;
                        index = i;
                    }
                }
            }
        }

        [BurstCompile]
        public static int ComputeNodeParent(ref KDTreeHeader header, ref KDTreeData data, int depth, int nodeIndexAtDepth)
        {
            return (1 << depth) - 1 + nodeIndexAtDepth;
        }

        [BurstCompile]
        public static void ComputeNodeIndexRange(out int left, out int right, ref KDTreeHeader header, ref KDTreeData data, int nodeIndex)
        {
            left = data.nodeRanges[nodeIndex].x;
            right = data.nodeRanges[nodeIndex].y;
        }
        
        // http://jcgt.org/published/0002/01/03/paper.pdf
        [BurstCompile]
        public struct KDTreeTraversalStateGeneralizedOptimized
        {
            public uint node;
            public uint traversal;

            public static readonly KDTreeTraversalStateGeneralizedOptimized zero = new KDTreeTraversalStateGeneralizedOptimized
            {
                node = 1,
                traversal = 0
            };
        }

        [BurstCompile]
        public static bool FindNearestNeighborStacklessGeneralizedIteratorOptimized(out int left, out int right, float distanceSquared, ref KDTreeTraversalStateGeneralizedOptimized state, ref KDTreeHeader header,
            ref KDTreeData data, float3 position)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);
            Debug.Assert(header.depthCount > 0);

            left = -1;
            right = -1;
            
            while (state.node > 0)
            {
                int nodeIndex = (int)(state.node - 1u);

                int depth = (state.node == 0) ? 0 : (int)FloorLog2((uint)state.node);

                if (IsLeaf(ref header, ref data, (int)depth))
                {
                    // Leaf node.
                    // Return range to loop over all contained elements and find nearest point.
                    // Perform an AABB check here to early out.
                    // Previous aabb checks during the traversal simply test which side of the split plane we are on.
                    float distance = (distanceSquared > 1e-5f) ? math.sqrt(distanceSquared) : 0.0f;
                    float3 aabbMin = data.aabbs[nodeIndex * 2 + 0];
                    float3 aabbMax = data.aabbs[nodeIndex * 2 + 1];
                    if (ComputeAABBContainsSphere(aabbMin, aabbMax, position, distance))
                    {
                        {
                            left = data.nodeRanges[nodeIndex].x;
                            right = data.nodeRanges[nodeIndex].y;
                        }

                        {
                            ++state.traversal;
                            int up = (int)CountTrailingZeros(state.traversal);
                            state.traversal >>= up;
                            state.node >>= up;

                            bool isDone = state.node <= 1;

                            if (isDone)
                            {
                                state.node = 0; // force the next iteration of the loop to exit.
                            }
                            else
                            {
                                state.node = state.node + 1 - ((state.node & 1) << 1); // sibling
                            }
                        }
                        return true;
                    }
                }
                else
                {
                    bool isAnyChildrenAccepted = false;
                    bool isRightChildFirst = false;
                    bool isRejectOneChild = false;
                    
                    // Branch
                    // Node has children.
                    // Recurse into nearest child.
                    // (1 << 0) == 1
                    // (1 << 1) == 2
                    // (1 << 2) == 4
                    // (1 << 4) == 8
                    //
                    //          0
                    //    1            2
                    //  3   4       5     6
                    // 7 8 9 10   11 12 13 14
                    int sortAxisParent = data.splitAxes[nodeIndex];
                    float splitPlaneParent = data.splitPositions[nodeIndex];
                    float splitPlaneDistanceSigned = position[sortAxisParent] - splitPlaneParent;

                    int childNodeIndexAtDepthLeft = (int)(state.node << 1) - 1;
                    int childNodeIndexAtDepthRight = childNodeIndexAtDepthLeft + 1;

                    // Use split plane as heuristic for deciding which child to enter first.
                    isRightChildFirst = splitPlaneDistanceSigned > 0.0f;

                    // Use actual AABBs to determine whether or not we need to enter the child.
                    float distance = (distanceSquared > 1e-5f) ? math.sqrt(distanceSquared) : 0.0f;
                    float3 sampleMin = position - new float3(distance, distance, distance);
                    float3 sampleMax = position + new float3(distance, distance, distance);
                    {

                        int nodeIndexChild = childNodeIndexAtDepthLeft;
                        float3 aabbMin = data.aabbs[nodeIndexChild * 2 + 0];
                        float3 aabbMax = data.aabbs[nodeIndexChild * 2 + 1];

                        if (ComputeAABBContainsAABB(aabbMin, aabbMax, sampleMin, sampleMax))
                        {
                            isAnyChildrenAccepted = true;
                        }
                        else
                        {
                            isRejectOneChild = true;
                        }
                    }

                    {

                        int nodeIndexChild = childNodeIndexAtDepthRight;
                        float3 aabbMin = data.aabbs[nodeIndexChild * 2 + 0];
                        float3 aabbMax = data.aabbs[nodeIndexChild * 2 + 1];

                        if (ComputeAABBContainsAABB(aabbMin, aabbMax, sampleMin, sampleMax))
                        {
                            isAnyChildrenAccepted = true;
                        }
                        else
                        {
                            isRejectOneChild = true;
                        }
                    }
			
                    if (isAnyChildrenAccepted)
                    {
                        state.node += state.node + (isRightChildFirst ? 1u : 0u);
                        state.traversal += state.traversal + (isRejectOneChild ? 1u : 0u);
                        continue;
                    }
                }

                {
                    ++state.traversal;
                    int up = (int)CountTrailingZeros(state.traversal);

                    state.traversal >>= up;
                    state.node >>= up;
		
                    bool isDone = state.node <= 1;
                    if (isDone)
                    {
                        return false;
                    }

                    state.node = state.node + 1 - ((state.node & 1) << 1); // sibling
                }
            }

            return false;
        }
        
        [BurstCompile]
        public static bool FindRayIntersectionStacklessGeneralizedIteratorOptimized(out int left, out int right, float distanceSquared, ref KDTreeTraversalStateGeneralizedOptimized state, ref KDTreeHeader header,
            ref KDTreeData data, float3 rayOrigin, float3 rayDirectionInverse)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);
            Debug.Assert(header.depthCount > 0);

            left = -1;
            right = -1;
            
            while (state.node > 0)
            {
                int nodeIndex = (int)(state.node - 1u);

                int depth = (state.node == 0) ? 0 : (int)FloorLog2((uint)state.node);

                if (IsLeaf(ref header, ref data, (int)depth))
                {
                    // Leaf node.
                    // Return range to loop over all contained elements and find nearest point.
                    // Perform an AABB check here to early out.
                    // Previous aabb checks during the traversal simply test which side of the split plane we are on.
                    float distance = (distanceSquared > 1e-5f) ? math.sqrt(distanceSquared) : 0.0f;
                    float3 aabbMin = data.aabbs[nodeIndex * 2 + 0];
                    float3 aabbMax = data.aabbs[nodeIndex * 2 + 1];
                    if (ComputeAABBContainsSphere(aabbMin, aabbMax, rayOrigin, distance))
                    {
                        {
                            left = data.nodeRanges[nodeIndex].x;
                            right = data.nodeRanges[nodeIndex].y;
                        }

                        // Leaf node found, inline traversal state update before returning
                        {
                            ++state.traversal;
                            int up = (int)CountTrailingZeros(state.traversal);
                            state.traversal >>= up;
                            state.node >>= up;

                            bool isDone = state.node <= 1;

                            if (isDone)
                            {
                                state.node = 0; // force the next iteration of the loop to exit.
                            }
                            else
                            {
                                state.node = state.node + 1 - ((state.node & 1) << 1); // sibling
                            }
                        }
                        return true;
                    }
                }
                else
                {
                    bool isAnyChildrenAccepted = false;
                    bool isRightChildFirst = false;
                    bool isRejectOneChild = false;
                    
                    // Branch
                    // Node has children.
                    // Recurse into nearest child.
                    // (1 << 0) == 1
                    // (1 << 1) == 2
                    // (1 << 2) == 4
                    // (1 << 4) == 8
                    //
                    //          0
                    //    1            2
                    //  3   4       5     6
                    // 7 8 9 10   11 12 13 14
                    int childNodeIndexLeft = (int)(state.node << 1) - 1;
                    int childNodeIndexRight = childNodeIndexLeft + 1;

                    float distance = (distanceSquared > 1e-5f) ? math.sqrt(distanceSquared) : 0.0f;
                    float childLeftT0 = 0.0f;
                    float childLeftT1 = distance;
                    {
                        float3 aabbMin = data.aabbs[childNodeIndexLeft * 2 + 0];
                        float3 aabbMax = data.aabbs[childNodeIndexLeft * 2 + 1];

                        if (IntersectRayAABB(ref childLeftT0, ref childLeftT1, rayOrigin, rayDirectionInverse, aabbMin, aabbMax))
                        {
                            isAnyChildrenAccepted = true;
                        }
                        else
                        {
                            isRejectOneChild = true;
                        }
                    }
                    
                    float childRightT0 = 0.0f;
                    float childRightT1 = distance;
                    {
                        float3 aabbMin = data.aabbs[childNodeIndexRight * 2 + 0];
                        float3 aabbMax = data.aabbs[childNodeIndexRight * 2 + 1];
                        
                        if (IntersectRayAABB(ref childRightT0, ref childRightT1, rayOrigin, rayDirectionInverse, aabbMin, aabbMax))
                        {
                            isAnyChildrenAccepted = true;
                            isRightChildFirst = isRejectOneChild
                                ? true
                                : ((childLeftT0 == childRightT0)
                                    ? (childRightT1 < childLeftT1)
                                    : (childRightT0 < childLeftT0));
                        }
                        else
                        {
                            isRejectOneChild = true;
                        }
                    }

                    if (isAnyChildrenAccepted)
                    {
                        state.node += state.node + (isRightChildFirst ? 1u : 0u);
                        state.traversal += state.traversal + (isRejectOneChild ? 1u : 0u);
                        continue;
                    }
                }

                

                {
                    ++state.traversal;
                    int up = (int)CountTrailingZeros(state.traversal);

                    state.traversal >>= up;
                    state.node >>= up;
		
                    bool isDone = state.node <= 1;
                    if (isDone)
                    {
                        return false;
                    }

                    state.node = state.node + 1 - ((state.node & 1) << 1); // sibling
                }
            }

            return false;
        }
        
        // https://jcgt.org/published/0002/02/02/
        [BurstCompile]
        private static bool IntersectRayAABB( ref float tmin, ref float tmax, float3 rayOrigin, float3 rayDirectionInverse, float3 aabbMin, float3 aabbMax)
        {
            float txmin, txmax, tymin, tymax, tzmin, tzmax;
            txmin = (((rayDirectionInverse.x < 0) ? aabbMax.x : aabbMin.x) - rayOrigin.x) * rayDirectionInverse.x;
            txmax = (((rayDirectionInverse.x < 0) ? aabbMin.x : aabbMax.x) - rayOrigin.x) * rayDirectionInverse.x;
            tymin = (((rayDirectionInverse.y < 0) ? aabbMax.y : aabbMin.y) - rayOrigin.y) * rayDirectionInverse.y;
            tymax = (((rayDirectionInverse.y < 0) ? aabbMin.y : aabbMax.y) - rayOrigin.y) * rayDirectionInverse.y;
            tzmin = (((rayDirectionInverse.z < 0) ? aabbMax.z : aabbMin.z) - rayOrigin.z) * rayDirectionInverse.z;
            tzmax = (((rayDirectionInverse.z < 0) ? aabbMin.z : aabbMax.z) - rayOrigin.z) * rayDirectionInverse.z;
            
            tmin = math.max(tzmin, math.max(tymin, math.max(txmin, tmin)));
            tmax = math.min(tzmax, math.min(tymax, math.min(txmax, tmax)));
            tmax *= 1.00000024f;
            return tmin <= tmax;
        }
        
        [BurstCompile]
        // Branchless implementation
        public static bool IntersectRayTriangle(
            out float t,
            out float3 triangleFaceNormal,
            out float triangleBarycentricA,
            out float triangleBarycentricB,
            out float triangleBarycentricC,
            float3 rayOrigin, float3 rayDirection, float tmin, float tmax,
            float3 triangleVertexPositionA, float3 triangleVertexPositionB, float3 triangleVertexPositionC
        )
        {
            float3 e0 = triangleVertexPositionB - triangleVertexPositionA;
            float3 e1 = triangleVertexPositionA - triangleVertexPositionC;
            float3 n  = math.cross(e1, e0);
            triangleFaceNormal = math.normalizesafe(n, new float3(0.0f, 1.0f, 0.0f));

            float3 e2 = (1.0f / math.dot( n, rayDirection)) * (triangleVertexPositionA - rayOrigin);
            float3 i  = math.cross(rayDirection, e2 );

            triangleBarycentricB  = math.dot( i, e1 );
            triangleBarycentricC = math.dot( i, e0 );
            triangleBarycentricA = 1.0f - triangleBarycentricB - triangleBarycentricC;
            t = math.dot( n, e2 );

            return
                (t < tmax)
                && (t > tmin) 
                && (triangleBarycentricB >= 0.0f)
                && (triangleBarycentricC >= 0.0f)
                && ((triangleBarycentricB + triangleBarycentricC) <= 1.0f);
        }

        [BurstCompile]
        private static bool IsNodeLeft(int nodeIndexAtDepth)
        {
            return (((uint)nodeIndexAtDepth) & 1u) == 0;
        }

        [BurstCompile]
        private static bool ComputeAABBContainsPosition(float3 aabbMin, float3 aabbMax, float3 position)
        {
            return !(math.any(position < aabbMin) || math.any(position > aabbMax));
        }

        [BurstCompile]
        private static bool ComputeAABBContainsAABB(float3 aabbLHSMin, float3 aabbLHSMax, float3 aabbRHSMin, float3 aabbRHSMax)
        {
            return !(math.any(aabbRHSMax < aabbLHSMin) || math.any(aabbRHSMin > aabbLHSMax));
        }
        
        [BurstCompile]
        private static bool ComputeAABBContainsSphere(float3 aabbMin, float3 aabbMax, float3 spherePosition, float sphereRadius)
        {
            float distanceSquared = 0.0f;
            for( int i = 0; i < 3; i++ )
            {
                float v = spherePosition[i];
                if( v < aabbMin[i] ) distanceSquared += (aabbMin[i] - v) * (aabbMin[i] - v);
                if( v > aabbMax[i] ) distanceSquared += (v - aabbMax[i]) * (v - aabbMax[i]);
            }

            return distanceSquared <= (sphereRadius * sphereRadius);
        }

        [BurstCompile]
        public static float4 ComputeDebugColorFromNodeIndex(int nodeIndex, int nodeCount)
        {
            // Spiral through chroma space.
            // Constant luminance.

            float radiusNormalized = ((float)nodeIndex + 0.5f) / (float)nodeCount;
            float theta = radiusNormalized * 2.0f * math.PI;
            float radius = radiusNormalized; // chroma space is in [0.0f, 1.0] range.
            float2 chroma = new float2(
                math.cos(theta),
                math.sin(theta)
            ) * radius;

            chroma += 0.5f;

            float3 ycocg = new float3(1.0f, chroma);
            float3 rgb;
            rgb.y = ycocg.x + ycocg.z;
            float temp = ycocg.x - ycocg.z;
            rgb.x = temp + ycocg.y;
            rgb.z = temp - ycocg.y;

            return new float4(rgb, 1.0f);
        }

        [BurstCompile]
        public static int ComputeDepthCountFromElements(int elementCount, int elementsPerLeaf)
        {
            return math.max(1, CeilLog2((elementCount + elementsPerLeaf - 1) / elementsPerLeaf));
        }

        // Basic implementation of ceil(log2(x)) which does not handle case of x == 0.
        [BurstCompile]
        private static uint CeilLog2(uint x)
        {
            Debug.Assert(x > 0);
            uint i = 0;
            uint y = x;
            if (y >= (1u << 16)) { i += 16u; y >>= 16; }
            if (y >= (1u << 8)) { i += 8u; y >>= 8; }
            if (y >= (1u << 4)) { i += 4u; y >>= 4; }
            if (y >= (1u << 2)) { i += 2u; y >>= 2; }
            if (y >= (1u << 1)) { i += 1u; y >>= 1; }
            return (x > (1u << (int)i)) ? (i + 1u) : i;
        }
        
        [BurstCompile]
        private static uint FloorLog2(uint x)
        {
            Debug.Assert(x > 0);
            uint i = 0;
            uint y = x;
    
            if (y >= (1u << 16)) { i += 16u; y >>= 16; }
            if (y >= (1u << 8)) { i += 8u; y >>= 8; }
            if (y >= (1u << 4)) { i += 4u; y >>= 4; }
            if (y >= (1u << 2)) { i += 2u; y >>= 2; }
            if (y >= (1u << 1)) { i += 1u; }

            return i;
        }

        [BurstCompile]
        private static int CeilLog2(int x)
        {
            Debug.Assert(x >= 0);

            return (int)CeilLog2((uint)x);
        }

        // Hackers Delight Figure 5-14
        [BurstCompile]
        private static uint CountTrailingZeros(uint i)
        {
            uint y;
            if (i == 0) { return 32; }
            uint n = 31;
            y = i << 16; if (y != 0) { n = n - 16; i = y; }
            y = i << 8; if (y != 0) { n = n - 8; i = y; }
            y = i << 4; if (y != 0) { n = n - 4; i = y; }
            y = i << 2; if (y != 0) { n = n - 2; i = y; }
            return n - ((i << 1) >> 31);
        }
    }
}