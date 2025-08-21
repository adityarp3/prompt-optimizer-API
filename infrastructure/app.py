#!/usr/bin/env python3
"""
AWS CDK Infrastructure for Prompt Optimizer API
This creates a production-ready ECS Fargate deployment with:
- Application Load Balancer
- Auto Scaling
- CloudWatch monitoring
- ECR repository
- VPC with public/private subnets
"""

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_ecr as ecr,
    aws_logs as logs,
    aws_iam as iam,
    aws_route53 as route53,
    aws_certificatemanager as acm,
    aws_applicationautoscaling as appautoscaling,
    Duration
)
from constructs import Construct
import os


class PromptOptimizerStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ECR Repository for Docker images
        ecr_repository = ecr.Repository(
            self, "PromptOptimizerECR",
            repository_name="prompt-optimizer-api",
            image_scan_on_push=True,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Keep only 10 images",
                    max_image_count=10
                )
            ]
        )

        # VPC with public and private subnets
        vpc = ec2.Vpc(
            self, "PromptOptimizerVPC",
            max_azs=2,
            nat_gateways=1,  # Cost optimization - use 1 NAT gateway
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                )
            ]
        )

        # ECS Cluster
        cluster = ecs.Cluster(
            self, "PromptOptimizerCluster",
            cluster_name="prompt-optimizer-cluster",
            vpc=vpc,
            container_insights=True
        )

        # CloudWatch Log Group
        log_group = logs.LogGroup(
            self, "PromptOptimizerLogs",
            log_group_name="/ecs/prompt-optimizer-api",
            retention=logs.RetentionDays.TWO_WEEKS
        )

        # Task Definition
        task_definition = ecs.FargateTaskDefinition(
            self, "PromptOptimizerTaskDef",
            memory_limit_mib=1024,
            cpu=512
        )

        # Container Definition
        container = task_definition.add_container(
            "prompt-optimizer-container",
            image=ecs.ContainerImage.from_ecr_repository(ecr_repository, "latest"),
            memory_limit_mib=1024,
            cpu=512,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="prompt-optimizer",
                log_group=log_group
            ),
            environment={
                "LOG_LEVEL": "INFO",
                "PYTHONPATH": "/app"
            },
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                retries=3,
                start_period=Duration.seconds(60)
            )
        )

        container.add_port_mappings(
            ecs.PortMapping(
                container_port=8000,
                protocol=ecs.Protocol.TCP
            )
        )

        # Application Load Balanced Fargate Service
        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "PromptOptimizerService",
            cluster=cluster,
            task_definition=task_definition,
            service_name="prompt-optimizer-service",
            public_load_balancer=True,
            listener_port=80,
            memory_limit_mib=1024,
            cpu=512,
            desired_count=2,
            # domain_name="api.yourdomain.com",  # Uncomment if you have a domain
            # domain_zone=route53.HostedZone.from_lookup(self, "Zone", domain_name="yourdomain.com"),
            platform_version=ecs.FargatePlatformVersion.LATEST
        )

        # Configure health checks
        fargate_service.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200"
        )

        # Auto Scaling
        scaling = fargate_service.service.auto_scale_task_count(
            min_capacity=1,
            max_capacity=10
        )

        scaling.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.minutes(5),
            scale_out_cooldown=Duration.minutes(2)
        )

        scaling.scale_on_memory_utilization(
            "MemoryScaling",
            target_utilization_percent=80,
            scale_in_cooldown=Duration.minutes(5),
            scale_out_cooldown=Duration.minutes(2)
        )

        # CloudWatch Alarms
        fargate_service.service.metric_cpu_utilization().create_alarm(
            self, "HighCpuAlarm",
            threshold=80,
            evaluation_periods=2
        )

        fargate_service.service.metric_memory_utilization().create_alarm(
            self, "HighMemoryAlarm",
            threshold=85,
            evaluation_periods=2
        )

        # Outputs
        cdk.CfnOutput(
            self, "LoadBalancerDNS",
            value=fargate_service.load_balancer.load_balancer_dns_name,
            description="Load Balancer DNS name"
        )

        cdk.CfnOutput(
            self, "ECRRepositoryURI",
            value=ecr_repository.repository_uri,
            description="ECR Repository URI"
        )

        cdk.CfnOutput(
            self, "ECSClusterName",
            value=cluster.cluster_name,
            description="ECS Cluster name"
        )

        cdk.CfnOutput(
            self, "ECSServiceName",
            value=fargate_service.service.service_name,
            description="ECS Service name"
        )


# CDK App
app = cdk.App()
PromptOptimizerStack(app, "PromptOptimizerStack",
    # Automatically use the account and region from your AWS credentials
    env=cdk.Environment(
        account=os.environ.get('CDK_DEFAULT_ACCOUNT'),
        region=os.environ.get('CDK_DEFAULT_REGION', 'us-east-1')
    )
)

app.synth()