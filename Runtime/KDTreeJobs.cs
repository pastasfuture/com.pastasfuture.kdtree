using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;

namespace Pastasfuture.KDTree.Runtime
{
    public static class KDTreeJobs
    {
        [BurstCompile]
        public struct TryAddPositionsAndRadiusConstantJob : IJob
        {
            // In order to work around current limitations of the dependancy graph in job system (and copy-style structs),
            // we supply a NativeArray<KDTree.KDTreeHeader> of Length 1.
            // This allows the header struct to be updated in the job, and passed to dependant jobs.
            public NativeArray<KDTree.KDTreeHeader> headerForJobs;
            public KDTree.KDTreeData data;

            [ReadOnly] public NativeArray<float3> positions;
            [ReadOnly] public float radius;

            public void Execute()
            {
                KDTree.KDTreeHeader header = headerForJobs[0];

                Debug.Assert(KDTree.Capacity(ref header, ref data) >= (header.count + positions.Length));

                for (int i = 0, iLen = positions.Length; i < iLen; ++i)
                {
                    float3 position = positions[i];
                    bool added = KDTree.TryAdd(ref header, ref data, i, position, radius);
                    if (!added)
                    {
                        // WARNING: Working around a burst compilation bug here.
                        // Previously I simply had this code, with no if statement:
                        // Debug.Assert(added);
                        // It looks like if we do not operate on this 'added' variable, it is stripped out in release mode,
                        // which seems to make my assert silently fail, and causes nothing to be added to the KDTree.
                        // By branching here, it forces the compiler to keep the 'added' variable around.
                        Debug.Assert(false);
                        break;
                    }
                }
                
                headerForJobs[0] = header;
            }
        }

        [BurstCompile]
        public struct TryAddPositionsAndRadiiJob : IJob
        {
            // In order to work around current limitations of the dependancy graph in job system (and copy-style structs),
            // we supply a NativeArray<KDTree.KDTreeHeader> of Length 1.
            // This allows the header struct to be updated in the job, and passed to dependant jobs.
            public NativeArray<KDTree.KDTreeHeader> headerForJobs;
            public KDTree.KDTreeData data;

            [ReadOnly] public NativeArray<float3> positions;
            [ReadOnly] public NativeArray<float> radii;

            public void Execute()
            {
                KDTree.KDTreeHeader header = headerForJobs[0];

                Debug.Assert(KDTree.Capacity(ref header, ref data) >= (header.count + positions.Length));

                for (int i = 0, iLen = positions.Length; i < iLen; ++i)
                {
                    float3 position = positions[i];
                    float radius = radii[i];
                    bool added = KDTree.TryAdd(ref header, ref data, i, position, radius);
                    if (!added)
                    {
                        // WARNING: Working around a burst compilation bug here.
                        // Previously I simply had this code, with no if statement:
                        // Debug.Assert(added);
                        // It looks like if we do not operate on this 'added' variable, it is stripped out in release mode,
                        // which seems to make my assert silently fail, and causes nothing to be added to the KDTree.
                        // By branching here, it forces the compiler to keep the 'added' variable around.
                        Debug.Assert(false);
                        break;
                    }
                }

                headerForJobs[0] = header;
            }
        }
        
        [BurstCompile]
        public struct TryAddPositionsAndRadiiWithCountJob : IJob
        {
            // In order to work around current limitations of the dependancy graph in job system (and copy-style structs),
            // we supply a NativeArray<KDTree.KDTreeHeader> of Length 1.
            // This allows the header struct to be updated in the job, and passed to dependant jobs.
            public NativeArray<KDTree.KDTreeHeader> headerForJobs;
            public KDTree.KDTreeData data;

            [ReadOnly] public NativeArray<float3> positions;
            [ReadOnly] public NativeArray<float> radii;
            [ReadOnly] public int positionsCount;

            public void Execute()
            {
                KDTree.KDTreeHeader header = headerForJobs[0];

                Debug.Assert(KDTree.Capacity(ref header, ref data) >= (header.count + positionsCount));

                for (int i = 0, iLen = positionsCount; i < iLen; ++i)
                {
                    float3 position = positions[i];
                    float radius = radii[i];
                    bool added = KDTree.TryAdd(ref header, ref data, i, position, radius);
                    if (!added)
                    {
                        // WARNING: Working around a burst compilation bug here.
                        // Previously I simply had this code, with no if statement:
                        // Debug.Assert(added);
                        // It looks like if we do not operate on this 'added' variable, it is stripped out in release mode,
                        // which seems to make my assert silently fail, and causes nothing to be added to the KDTree.
                        // By branching here, it forces the compiler to keep the 'added' variable around.
                        Debug.Assert(false);
                        break;
                    }
                }

                headerForJobs[0] = header;
            }
        }

        [BurstCompile]
        public struct BuildJob : IJob
        {
            // In order to work around current limitations of the dependancy graph in job system (and copy-style structs),
            // we supply a NativeArray<KDTree.KDTreeHeader> of Length 1.
            // This allows the header struct to be updated in the job, and passed to dependant jobs.
            public NativeArray<KDTree.KDTreeHeader> headerForJobs;
            public KDTree.KDTreeData data;

            [ReadOnly] public int depthCount;

            public void Execute()
            {
                KDTree.KDTreeHeader header = headerForJobs[0];
                KDTree.Build(ref header, ref data, depthCount);
                headerForJobs[0] = header;
            }
        }

        [BurstCompile]
        public struct FindNearestNeighborsJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<KDTree.KDTreeHeader> headerForJobs;
            [ReadOnly] public KDTree.KDTreeData data;
            [ReadOnly] public NativeArray<float3> queryPositions;

            public NativeArray<int> resIndices;
            public NativeArray<float> resDistances;

            public void Execute(int queryPositionIndex)
            {
                float3 queryPosition = queryPositions[queryPositionIndex];

                KDTree.KDTreeHeader header = headerForJobs[0];
                KDTree.FindNearestNeighbor(out int index, out float distanceSquared, ref header, ref data, queryPosition);

                resIndices[queryPositionIndex] = (index >= 0) ? data.payloadIndices[index] : -1;
                resDistances[queryPositionIndex] = (distanceSquared > 1e-5f) ? math.sqrt(distanceSquared) : 0.0f;
            }
        }
    }
}