"""
Générateur de Dataset - Logs CI/CD (Flaky vs Non-Flaky)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

np.random.seed(42)

COMPONENTS = ['auth', 'payment', 'api', 'database', 'cache', 'queue', 
              'storage', 'notification', 'search', 'analytics', 'user', 'session']
ENVIRONMENTS = ['dev', 'staging', 'prod', 'test', 'integration']
SEVERITIES = ['info', 'warning', 'error', 'critical']
BUILD_TOOLS = ['maven', 'gradle', 'npm', 'pip', 'cargo', 'go']
OS_LIST = ['linux', 'windows', 'macos']
LANGUAGES = ['java', 'python', 'javascript', 'go', 'rust', 'kotlin']
CI_PLATFORMS = ['jenkins', 'github_actions', 'gitlab_ci', 'circleci', 'travis']
TRIGGER_TYPES = ['push', 'pull_request', 'schedule', 'manual', 'tag']
BRANCH_TYPES = ['main', 'develop', 'feature', 'hotfix', 'release']
DAYS_OF_WEEK = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

FLAKY_ERROR_TYPES = [
    'TimeoutException', 'ConnectionResetException', 'SocketTimeoutException',
    'ResourceBusyException', 'LockContentionException', 'NetworkException',
    'TransientFailureException', 'ServiceUnavailableException', 'RetryableException',
    'ConcurrencyException', 'RaceConditionException', 'TemporaryFailureException'
]

NON_FLAKY_ERROR_TYPES = [
    'NullPointerException', 'IndexOutOfBoundsException', 'IllegalArgumentException',
    'ClassCastException', 'ArithmeticException', 'AssertionError',
    'ValidationException', 'IllegalStateException', 'SecurityException',
    'ConfigurationException', 'MissingFieldException', 'TypeMismatchException'
]

FLAKY_LOG_TEMPLATES = [
    "Connection to {service} timed out after {timeout}ms, retry attempt {retry}",
    "Service {service} temporarily unavailable, backing off for {backoff}ms",
    "Network latency spike detected: {latency}ms, threshold exceeded",
    "Resource {resource} busy, waiting for lock release",
    "Socket connection reset by peer during {operation}",
    "Database connection pool exhausted, waiting for available connection",
    "External API {service} returned 503, scheduled retry in {backoff}ms",
    "Test {test} passed after {retry} retries due to timing issue",
    "Intermittent failure in {component}: {message}",
    "Race condition detected in concurrent test execution",
]

NON_FLAKY_LOG_TEMPLATES = [
    "NullPointerException at {file}:{line} - {variable} is null",
    "Array index {index} out of bounds for length {length}",
    "Invalid argument: {param} cannot be {value}",
    "Type mismatch: expected {expected} but got {actual}",
    "Division by zero in {method} at {file}:{line}",
    "Assertion failed: {condition} was false",
    "Missing required field '{field}' in {object}",
    "Configuration error: property '{property}' not found",
    "Security violation: unauthorized access to {resource}",
    "Circular dependency detected in {component}",
]

FLAKY_STACK_TRACES = [
    """java.net.SocketTimeoutException: Read timed out
    at java.net.SocketInputStream.socketRead0(Native Method)
    at com.example.{component}.{class}.{method}({class}.java:{line})""",
    """org.apache.http.conn.ConnectTimeoutException: Connect to {host} timed out
    at org.apache.http.impl.conn.DefaultHttpClientConnectionOperator.connect
    at com.example.{component}.HttpClient.execute(HttpClient.java:{line})""",
]

FLAKY_STACK_TRACES_LONG = [
    """java.net.SocketTimeoutException: Read timed out after 30000ms waiting for response from {host}
    at java.base/java.net.SocketInputStream.socketRead0(Native Method)
    at java.base/java.net.SocketInputStream.socketRead(SocketInputStream.java:115)
    at java.base/java.net.SocketInputStream.read(SocketInputStream.java:168)
    at java.base/java.net.SocketInputStream.read(SocketInputStream.java:140)
    at org.apache.http.impl.io.SessionInputBufferImpl.streamRead(SessionInputBufferImpl.java:137)
    at org.apache.http.impl.io.SessionInputBufferImpl.fillBuffer(SessionInputBufferImpl.java:153)
    at org.apache.http.impl.io.SessionInputBufferImpl.readLine(SessionInputBufferImpl.java:280)
    at org.apache.http.impl.conn.DefaultHttpResponseParser.parseHead(DefaultHttpResponseParser.java:138)
    at org.apache.http.impl.conn.DefaultHttpResponseParser.parseHead(DefaultHttpResponseParser.java:56)
    at org.apache.http.impl.io.AbstractMessageParser.parse(AbstractMessageParser.java:259)
    at org.apache.http.impl.DefaultBHttpClientConnection.receiveResponseHeader(DefaultBHttpClientConnection.java:163)
    at org.apache.http.impl.conn.CPoolProxy.receiveResponseHeader(CPoolProxy.java:157)
    at org.apache.http.protocol.HttpRequestExecutor.doReceiveResponse(HttpRequestExecutor.java:273)
    at org.apache.http.protocol.HttpRequestExecutor.execute(HttpRequestExecutor.java:125)
    at org.apache.http.impl.execchain.MainClientExec.execute(MainClientExec.java:272)
    at org.apache.http.impl.execchain.ProtocolExec.execute(ProtocolExec.java:186)
    at org.apache.http.impl.execchain.RetryExec.execute(RetryExec.java:89)
    at org.apache.http.impl.execchain.RedirectExec.execute(RedirectExec.java:110)
    at org.apache.http.impl.client.InternalHttpClient.doExecute(InternalHttpClient.java:185)
    at org.apache.http.impl.client.CloseableHttpClient.execute(CloseableHttpClient.java:83)
    at org.apache.http.impl.client.CloseableHttpClient.execute(CloseableHttpClient.java:108)
    at com.example.{component}.http.HttpClientWrapper.executeRequest(HttpClientWrapper.java:156)
    at com.example.{component}.http.HttpClientWrapper.doGet(HttpClientWrapper.java:89)
    at com.example.{component}.service.{class}Service.fetchExternalData({class}Service.java:{line})
    at com.example.{component}.service.{class}Service.processRequest({class}Service.java:{line2})
    at com.example.{component}.controller.{class}Controller.handleRequest({class}Controller.java:78)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:77)
    at java.base/jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
    at java.base/java.lang.reflect.Method.invoke(Method.java:568)
    at org.springframework.web.method.support.InvocableHandlerMethod.doInvoke(InvocableHandlerMethod.java:205)
    at org.springframework.web.method.support.InvocableHandlerMethod.invokeForRequest(InvocableHandlerMethod.java:150)
    at org.springframework.web.servlet.mvc.method.annotation.ServletInvocableHandlerMethod.invokeAndHandle(ServletInvocableHandlerMethod.java:117)
    at org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerAdapter.invokeHandlerMethod(RequestMappingHandlerAdapter.java:895)
    at org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerAdapter.handleInternal(RequestMappingHandlerAdapter.java:808)
    at org.springframework.web.servlet.mvc.method.AbstractHandlerMethodAdapter.handle(AbstractHandlerMethodAdapter.java:87)
    at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1067)
    at org.springframework.web.servlet.DispatcherServlet.doService(DispatcherServlet.java:963)
    at org.springframework.web.servlet.FrameworkServlet.processRequest(FrameworkServlet.java:1006)
    at org.springframework.web.servlet.FrameworkServlet.doGet(FrameworkServlet.java:898)
    at javax.servlet.http.HttpServlet.service(HttpServlet.java:655)
    at org.springframework.web.servlet.FrameworkServlet.service(FrameworkServlet.java:883)
    at org.springframework.test.web.servlet.TestDispatcherServlet.service(TestDispatcherServlet.java:72)
    at javax.servlet.http.HttpServlet.service(HttpServlet.java:764)
    at org.springframework.mock.web.MockFilterChain.doFilter(MockFilterChain.java:137)
    at org.springframework.test.web.servlet.MockMvc.perform(MockMvc.java:201)
    at com.example.{component}.test.{class}IntegrationTest.testExternalServiceConnection({class}IntegrationTest.java:145)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:77)
    at org.junit.platform.commons.util.ReflectionUtils.invokeMethod(ReflectionUtils.java:725)
    at org.junit.jupiter.engine.execution.MethodInvocation.proceed(MethodInvocation.java:60)
    at org.junit.jupiter.engine.execution.InvocationInterceptorChain.proceed(InvocationInterceptorChain.java:131)
    at org.junit.jupiter.engine.descriptor.TestMethodTestDescriptor.execute(TestMethodTestDescriptor.java:131)
    at org.junit.platform.engine.support.hierarchical.NodeTestTask.execute(NodeTestTask.java:151)
    at org.junit.platform.engine.support.hierarchical.ThrowableCollector.execute(ThrowableCollector.java:73)
    at org.junit.platform.engine.support.hierarchical.SameThreadHierarchicalTestExecutorService.submit(SameThreadHierarchicalTestExecutorService.java:35)
    at org.junit.platform.engine.support.hierarchical.HierarchicalTestExecutor.execute(HierarchicalTestExecutor.java:57)
    at org.junit.platform.launcher.core.EngineExecutionOrchestrator.execute(EngineExecutionOrchestrator.java:107)
    at org.junit.platform.launcher.core.DefaultLauncher.execute(DefaultLauncher.java:229)
    at org.apache.maven.surefire.junitplatform.JUnitPlatformProvider.execute(JUnitPlatformProvider.java:188)
    at org.apache.maven.surefire.booter.ForkedBooter.main(ForkedBooter.java:581)
Caused by: java.io.InterruptedIOException: Connection pool timeout reached waiting for connection from route {component}.internal.example.com -> {host}:443
    at org.apache.http.pool.AbstractConnPool.getPoolEntryBlocking(AbstractConnPool.java:392)
    at org.apache.http.pool.AbstractConnPool$2.get(AbstractConnPool.java:243)
    at org.apache.http.pool.AbstractConnPool$2.get(AbstractConnPool.java:188)
    at org.apache.http.impl.conn.PoolingHttpClientConnectionManager.leaseConnection(PoolingHttpClientConnectionManager.java:306)
    ... 58 more""",
    """org.apache.kafka.common.errors.TimeoutException: Topic {component}-events not present in metadata after 60000 ms
    at org.apache.kafka.clients.producer.KafkaProducer.waitOnMetadata(KafkaProducer.java:1073)
    at org.apache.kafka.clients.producer.KafkaProducer.doSend(KafkaProducer.java:961)
    at org.apache.kafka.clients.producer.KafkaProducer.send(KafkaProducer.java:921)
    at org.apache.kafka.clients.producer.KafkaProducer.send(KafkaProducer.java:807)
    at com.example.{component}.messaging.KafkaMessagePublisher.publishEvent(KafkaMessagePublisher.java:89)
    at com.example.{component}.messaging.KafkaMessagePublisher.publish(KafkaMessagePublisher.java:45)
    at com.example.{component}.service.EventService.emitEvent(EventService.java:156)
    at com.example.{component}.service.{class}Service.processAndNotify({class}Service.java:{line})
    at com.example.{component}.service.{class}Service.executeBusinessLogic({class}Service.java:{line2})
    at com.example.{component}.handler.{class}RequestHandler.handle({class}RequestHandler.java:78)
    at com.example.{component}.handler.{class}RequestHandler.handleRequest({class}RequestHandler.java:45)
    at com.example.core.dispatcher.RequestDispatcher.dispatch(RequestDispatcher.java:134)
    at com.example.core.dispatcher.RequestDispatcher.processIncoming(RequestDispatcher.java:89)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:77)
    at java.base/jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
    at java.base/java.lang.reflect.Method.invoke(Method.java:568)
    at org.springframework.aop.support.AopUtils.invokeJoinpointUsingReflection(AopUtils.java:344)
    at org.springframework.aop.framework.ReflectiveMethodInvocation.invokeJoinpoint(ReflectiveMethodInvocation.java:198)
    at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:163)
    at org.springframework.aop.interceptor.AsyncExecutionInterceptor.invoke(AsyncExecutionInterceptor.java:115)
    at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:186)
    at org.springframework.transaction.interceptor.TransactionInterceptor.invoke(TransactionInterceptor.java:119)
    at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:186)
    at org.springframework.aop.framework.JdkDynamicAopProxy.invoke(JdkDynamicAopProxy.java:215)
    at jdk.proxy2.$Proxy145.processIncoming(Unknown Source)
    at com.example.{component}.test.integration.{class}IntegrationTest.testEventPublishing({class}IntegrationTest.java:234)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at org.junit.platform.commons.util.ReflectionUtils.invokeMethod(ReflectionUtils.java:725)
    at org.junit.jupiter.engine.execution.MethodInvocation.proceed(MethodInvocation.java:60)
    at org.junit.jupiter.engine.execution.InvocationInterceptorChain.proceed(InvocationInterceptorChain.java:131)
    at org.junit.jupiter.engine.descriptor.TestMethodTestDescriptor.execute(TestMethodTestDescriptor.java:131)
    at org.junit.platform.launcher.core.DefaultLauncher.execute(DefaultLauncher.java:229)
    at org.apache.maven.surefire.booter.ForkedBooter.main(ForkedBooter.java:581)
Caused by: org.apache.kafka.common.errors.NetworkException: The server disconnected before a response was received
    at org.apache.kafka.clients.NetworkClient.handleDisconnection(NetworkClient.java:988)
    at org.apache.kafka.clients.NetworkClient.processDisconnection(NetworkClient.java:975)
    ... 34 more""",
]

NON_FLAKY_STACK_TRACES = [
    """java.lang.NullPointerException: Cannot invoke method on null object
    at com.example.{component}.{class}.{method}({class}.java:{line})
    at com.example.{component}.{class2}.{method2}({class2}.java:{line2})""",
    """java.lang.AssertionError: expected:<{expected}> but was:<{actual}>
    at org.junit.Assert.fail(Assert.java:89)
    at org.junit.Assert.assertEquals(Assert.java:118)""",
]

NON_FLAKY_STACK_TRACES_LONG = [
    """java.lang.NullPointerException: Cannot invoke "String.length()" because "this.{field}" is null
    at com.example.{component}.model.{class}Entity.validate({class}Entity.java:{line})
    at com.example.{component}.model.{class}Entity.prePersist({class}Entity.java:45)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:77)
    at java.base/jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
    at java.base/java.lang.reflect.Method.invoke(Method.java:568)
    at org.hibernate.jpa.event.internal.EntityCallback.performCallback(EntityCallback.java:51)
    at org.hibernate.jpa.event.internal.CallbackRegistryImpl.callback(CallbackRegistryImpl.java:98)
    at org.hibernate.jpa.event.internal.CallbackRegistryImpl.preCreate(CallbackRegistryImpl.java:50)
    at org.hibernate.event.internal.AbstractSaveEventListener.performSaveOrReplicate(AbstractSaveEventListener.java:241)
    at org.hibernate.event.internal.AbstractSaveEventListener.performSave(AbstractSaveEventListener.java:167)
    at org.hibernate.event.internal.AbstractSaveEventListener.saveWithGeneratedId(AbstractSaveEventListener.java:123)
    at org.hibernate.event.internal.DefaultPersistEventListener.entityIsTransient(DefaultPersistEventListener.java:185)
    at org.hibernate.event.internal.DefaultPersistEventListener.onPersist(DefaultPersistEventListener.java:128)
    at org.hibernate.event.internal.DefaultPersistEventListener.onPersist(DefaultPersistEventListener.java:55)
    at org.hibernate.event.service.internal.EventListenerGroupImpl.fireEventOnEachListener(EventListenerGroupImpl.java:107)
    at org.hibernate.internal.SessionImpl.firePersist(SessionImpl.java:755)
    at org.hibernate.internal.SessionImpl.persist(SessionImpl.java:738)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:77)
    at java.base/jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
    at java.base/java.lang.reflect.Method.invoke(Method.java:568)
    at org.springframework.orm.jpa.SharedEntityManagerCreator$SharedEntityManagerInvocationHandler.invoke(SharedEntityManagerCreator.java:311)
    at jdk.proxy2.$Proxy123.persist(Unknown Source)
    at com.example.{component}.repository.{class}Repository.save({class}Repository.java:67)
    at com.example.{component}.repository.{class}Repository$$FastClassBySpringCGLIB$$abc123.invoke(<generated>)
    at org.springframework.cglib.proxy.MethodProxy.invoke(MethodProxy.java:218)
    at org.springframework.aop.framework.CglibAopProxy$CglibMethodInvocation.invokeJoinpoint(CglibAopProxy.java:793)
    at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:163)
    at org.springframework.aop.framework.CglibAopProxy$CglibMethodInvocation.proceed(CglibAopProxy.java:763)
    at org.springframework.transaction.interceptor.TransactionInterceptor.invoke(TransactionInterceptor.java:119)
    at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:186)
    at org.springframework.aop.framework.CglibAopProxy$DynamicAdvisedInterceptor.intercept(CglibAopProxy.java:708)
    at com.example.{component}.repository.{class}Repository$$EnhancerBySpringCGLIB$$def456.save(<generated>)
    at com.example.{component}.service.{class}Service.create{class}({class}Service.java:{line2})
    at com.example.{component}.service.{class}Service$$FastClassBySpringCGLIB$$ghi789.invoke(<generated>)
    at org.springframework.cglib.proxy.MethodProxy.invoke(MethodProxy.java:218)
    at com.example.{component}.controller.{class}Controller.create({class}Controller.java:89)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at org.springframework.web.method.support.InvocableHandlerMethod.doInvoke(InvocableHandlerMethod.java:205)
    at org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerAdapter.invokeHandlerMethod(RequestMappingHandlerAdapter.java:895)
    at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1067)
    at org.springframework.test.web.servlet.MockMvc.perform(MockMvc.java:201)
    at com.example.{component}.test.{class}ControllerTest.testCreate{class}WithNullField({class}ControllerTest.java:178)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at org.junit.platform.commons.util.ReflectionUtils.invokeMethod(ReflectionUtils.java:725)
    at org.junit.jupiter.engine.descriptor.TestMethodTestDescriptor.execute(TestMethodTestDescriptor.java:131)
    at org.junit.platform.launcher.core.DefaultLauncher.execute(DefaultLauncher.java:229)
    at org.apache.maven.surefire.booter.ForkedBooter.main(ForkedBooter.java:581)""",
    """java.lang.IllegalArgumentException: Invalid value for parameter '{param}': {value} is not allowed
    at com.example.{component}.validation.{class}Validator.validate({class}Validator.java:{line})
    at com.example.{component}.validation.{class}Validator.validateRequest({class}Validator.java:45)
    at com.example.{component}.service.{class}Service.processRequest({class}Service.java:123)
    at com.example.{component}.service.{class}Service.execute({class}Service.java:89)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:77)
    at java.base/jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
    at java.base/java.lang.reflect.Method.invoke(Method.java:568)
    at org.springframework.aop.support.AopUtils.invokeJoinpointUsingReflection(AopUtils.java:344)
    at org.springframework.aop.framework.ReflectiveMethodInvocation.invokeJoinpoint(ReflectiveMethodInvocation.java:198)
    at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:163)
    at org.springframework.validation.beanvalidation.MethodValidationInterceptor.invoke(MethodValidationInterceptor.java:123)
    at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:186)
    at org.springframework.transaction.interceptor.TransactionInterceptor.invoke(TransactionInterceptor.java:119)
    at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:186)
    at org.springframework.aop.framework.JdkDynamicAopProxy.invoke(JdkDynamicAopProxy.java:215)
    at jdk.proxy2.$Proxy156.execute(Unknown Source)
    at com.example.{component}.controller.{class}Controller.process({class}Controller.java:{line2})
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at org.springframework.web.method.support.InvocableHandlerMethod.doInvoke(InvocableHandlerMethod.java:205)
    at org.springframework.web.method.support.InvocableHandlerMethod.invokeForRequest(InvocableHandlerMethod.java:150)
    at org.springframework.web.servlet.mvc.method.annotation.ServletInvocableHandlerMethod.invokeAndHandle(ServletInvocableHandlerMethod.java:117)
    at org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerAdapter.invokeHandlerMethod(RequestMappingHandlerAdapter.java:895)
    at org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerAdapter.handleInternal(RequestMappingHandlerAdapter.java:808)
    at org.springframework.web.servlet.mvc.method.AbstractHandlerMethodAdapter.handle(AbstractHandlerMethodAdapter.java:87)
    at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1067)
    at org.springframework.web.servlet.DispatcherServlet.doService(DispatcherServlet.java:963)
    at org.springframework.web.servlet.FrameworkServlet.processRequest(FrameworkServlet.java:1006)
    at org.springframework.web.servlet.FrameworkServlet.doPost(FrameworkServlet.java:909)
    at javax.servlet.http.HttpServlet.service(HttpServlet.java:681)
    at org.springframework.web.servlet.FrameworkServlet.service(FrameworkServlet.java:883)
    at org.springframework.test.web.servlet.TestDispatcherServlet.service(TestDispatcherServlet.java:72)
    at javax.servlet.http.HttpServlet.service(HttpServlet.java:764)
    at org.springframework.mock.web.MockFilterChain.doFilter(MockFilterChain.java:137)
    at org.springframework.test.web.servlet.MockMvc.perform(MockMvc.java:201)
    at com.example.{component}.test.{class}ControllerTest.testInvalidParameter({class}ControllerTest.java:256)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at org.junit.platform.commons.util.ReflectionUtils.invokeMethod(ReflectionUtils.java:725)
    at org.junit.jupiter.engine.descriptor.TestMethodTestDescriptor.execute(TestMethodTestDescriptor.java:131)
    at org.junit.platform.launcher.core.DefaultLauncher.execute(DefaultLauncher.java:229)
    at org.apache.maven.surefire.booter.ForkedBooter.main(ForkedBooter.java:581)""",
]


def generate_log_message(is_flaky, component, retry_count):
    """Génère un message de log."""
    templates = FLAKY_LOG_TEMPLATES if is_flaky else NON_FLAKY_LOG_TEMPLATES
    template = np.random.choice(templates)
    
    vars_dict = {
        'service': np.random.choice(['redis', 'postgresql', 'mongodb', 'kafka', 'elasticsearch']),
        'timeout': np.random.choice([1000, 3000, 5000, 10000, 30000]),
        'retry': retry_count, 'backoff': np.random.choice([100, 500, 1000, 2000]),
        'latency': np.random.randint(100, 5000),
        'resource': np.random.choice(['connection', 'lock', 'file', 'memory']),
        'operation': np.random.choice(['read', 'write', 'connect', 'execute']),
        'test': f"Test{component.capitalize()}_{np.random.randint(1, 100)}",
        'component': component, 'message': np.random.choice(['resource unavailable', 'timeout exceeded']),
        'file': f"{component.capitalize()}Service.java", 'line': np.random.randint(10, 500),
        'variable': np.random.choice(['user', 'config', 'connection', 'result']),
        'index': np.random.randint(-5, 1000), 'length': np.random.randint(0, 100),
        'param': np.random.choice(['id', 'name', 'value', 'count']),
        'value': np.random.choice(['null', 'negative', 'empty']),
        'expected': np.random.choice(['200', 'true', 'SUCCESS']),
        'actual': np.random.choice(['500', 'false', 'ERROR', 'null']),
        'method': np.random.choice(['process', 'validate', 'execute', 'handle']),
        'condition': np.random.choice(['x > 0', 'result != null']),
        'field': np.random.choice(['id', 'name', 'timestamp', 'userId']),
        'object': np.random.choice(['Request', 'Response', 'Config']),
        'property': np.random.choice(['db.url', 'api.key', 'cache.ttl']),
    }
    
    result = template
    for key, val in vars_dict.items():
        result = result.replace('{' + key + '}', str(val))
    return result


def generate_stack_trace(is_flaky, component, test_name, use_long=False):
    """Génère une stack trace."""
    if np.random.random() < 0.3:
        return ""
    
    if use_long:
        templates = FLAKY_STACK_TRACES_LONG if is_flaky else NON_FLAKY_STACK_TRACES_LONG
    else:
        templates = FLAKY_STACK_TRACES if is_flaky else NON_FLAKY_STACK_TRACES
    template = np.random.choice(templates)
    
    vars_dict = {
        'component': component, 'class': f"{component.capitalize()}Service",
        'class2': f"{component.capitalize()}Handler",
        'method': np.random.choice(['process', 'execute', 'handle']),
        'method2': np.random.choice(['validate', 'transform']),
        'line': np.random.randint(50, 500), 'line2': np.random.randint(50, 500),
        'host': f"{component}.internal.example.com",
        'expected': np.random.choice(['200', 'true']),
        'actual': np.random.choice(['500', 'false', 'null']),
        'field': np.random.choice(['username', 'email', 'config', 'data']),
        'param': np.random.choice(['id', 'userId', 'orderId']),
        'value': np.random.choice(['null', '-1', 'empty']),
    }
    
    result = template
    for key, val in vars_dict.items():
        result = result.replace('{' + key + '}', str(val))
    return result


def generate_test_name(component):
    """Génère un nom de test."""
    actions = ['create', 'read', 'update', 'delete', 'validate', 'process', 'authenticate']
    objects = ['user', 'order', 'payment', 'session', 'config', 'data', 'request']
    return f"test_{np.random.choice(actions)}_{np.random.choice(objects)}_{component}"


def generate_sample(is_flaky, noise_probability=0.0, use_long_traces=False):
    """Génère un échantillon complet."""
    
    if np.random.random() < noise_probability:
        is_flaky = not is_flaky
    
    component = np.random.choice(COMPONENTS)
    environment = np.random.choice(ENVIRONMENTS)
    severity = np.random.choice(SEVERITIES)
    build_tool = np.random.choice(BUILD_TOOLS)
    os_val = np.random.choice(OS_LIST)
    language = np.random.choice(LANGUAGES)
    ci_platform = np.random.choice(CI_PLATFORMS)
    trigger_type = np.random.choice(TRIGGER_TYPES)
    branch_type = np.random.choice(BRANCH_TYPES)
    day_of_week = np.random.choice(DAYS_OF_WEEK)
    
    if is_flaky:
        is_timeout = np.random.random() < 0.45
        is_network_error = np.random.random() < 0.40
        has_retry = np.random.random() < 0.50
        parallel_execution = np.random.random() < 0.35
        has_external_dependency = np.random.random() < 0.45
        duration_ms = int(np.random.lognormal(7.5, 1.2))
        retry_count = np.random.choice([0, 1, 2, 3], p=[0.40, 0.30, 0.20, 0.10])
        failure_rate_history = np.clip(np.random.normal(0.45, 0.20), 0, 1)
        time_since_last_success = np.random.exponential(30)
        network_latency_ms = int(np.random.lognormal(4.5, 1.0))
        db_response_time_ms = int(np.random.lognormal(4, 0.8))
    else:
        is_timeout = np.random.random() < 0.30
        is_network_error = np.random.random() < 0.25
        has_retry = np.random.random() < 0.20
        parallel_execution = np.random.random() < 0.30
        has_external_dependency = np.random.random() < 0.35
        duration_ms = int(np.random.lognormal(7.0, 1.0))
        retry_count = np.random.choice([0, 1, 2, 3], p=[0.70, 0.20, 0.07, 0.03])
        failure_rate_history = np.clip(np.random.normal(0.30, 0.20), 0, 1)
        time_since_last_success = np.random.exponential(50)
        network_latency_ms = int(np.random.lognormal(4.0, 0.8))
        db_response_time_ms = int(np.random.lognormal(3.8, 0.7))
    
    is_first_run = np.random.random() < 0.2
    memory_mb = int(np.random.lognormal(6, 0.8))
    cpu_percent = min(100, int(np.random.lognormal(3, 0.8)))
    line_number = np.random.randint(1, 1000)
    file_count = np.random.randint(1, 50)
    test_count = np.random.randint(1, 500)
    queue_size = np.random.randint(0, 1000)
    
    test_name = generate_test_name(component)
    
    if np.random.random() < 0.30:
        error_type = np.random.choice(NON_FLAKY_ERROR_TYPES if is_flaky else FLAKY_ERROR_TYPES)
    else:
        error_type = np.random.choice(FLAKY_ERROR_TYPES if is_flaky else NON_FLAKY_ERROR_TYPES)
    
    if np.random.random() < 0.25:
        log_message = generate_log_message(not is_flaky, component, retry_count)
    else:
        log_message = generate_log_message(is_flaky, component, retry_count)
    
    if np.random.random() < 0.25:
        stack_trace = generate_stack_trace(not is_flaky, component, test_name, use_long=use_long_traces)
    else:
        stack_trace = generate_stack_trace(is_flaky, component, test_name, use_long=use_long_traces)
    
    return {
        'log_message': log_message, 'stack_trace': stack_trace,
        'test_name': test_name, 'error_type': error_type,
        'duration_ms': duration_ms, 'retry_count': retry_count,
        'memory_mb': memory_mb, 'cpu_percent': cpu_percent,
        'line_number': line_number, 'file_count': file_count,
        'test_count': test_count, 'failure_rate_history': round(failure_rate_history, 3),
        'time_since_last_success': round(time_since_last_success, 2),
        'network_latency_ms': network_latency_ms, 'db_response_time_ms': db_response_time_ms,
        'queue_size': queue_size, 'component': component, 'environment': environment,
        'severity': severity, 'build_tool': build_tool, 'os': os_val,
        'language': language, 'ci_platform': ci_platform, 'trigger_type': trigger_type,
        'branch_type': branch_type, 'day_of_week': day_of_week,
        'is_timeout': int(is_timeout), 'is_network_error': int(is_network_error),
        'has_retry': int(has_retry), 'is_first_run': int(is_first_run),
        'parallel_execution': int(parallel_execution),
        'has_external_dependency': int(has_external_dependency),
        'is_flaky': int(is_flaky)
    }


def generate_dataset(n_samples=5000, noise_ratio=0.1, seed=42, use_long_traces=False):
    """Génère le dataset complet."""
    np.random.seed(seed)
    
    samples = []
    n_flaky = n_samples // 2
    n_non_flaky = n_samples - n_flaky
    
    trace_info = " (avec longues stack traces)" if use_long_traces else ""
    print(f"Génération de {n_samples} échantillons (bruit: {noise_ratio*100:.0f}%){trace_info}...")
    
    for _ in range(n_flaky):
        samples.append(generate_sample(is_flaky=True, noise_probability=noise_ratio, use_long_traces=use_long_traces))
    for _ in range(n_non_flaky):
        samples.append(generate_sample(is_flaky=False, noise_probability=noise_ratio, use_long_traces=use_long_traces))
    
    np.random.shuffle(samples)
    df = pd.DataFrame(samples)
    
    column_order = [
        'log_message', 'stack_trace', 'test_name', 'error_type',
        'duration_ms', 'retry_count', 'memory_mb', 'cpu_percent', 'line_number',
        'file_count', 'test_count', 'failure_rate_history', 'time_since_last_success',
        'network_latency_ms', 'db_response_time_ms', 'queue_size',
        'component', 'environment', 'severity', 'build_tool', 'os', 'language',
        'ci_platform', 'trigger_type', 'branch_type', 'day_of_week',
        'is_timeout', 'is_network_error', 'has_retry', 'is_first_run',
        'parallel_execution', 'has_external_dependency', 'is_flaky'
    ]
    
    return df[column_order]


def main():
    parser = argparse.ArgumentParser(description='Génère un dataset de logs CI/CD')
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--long-traces', action='store_true', help='Utiliser des stack traces longues (>512 tokens)')
    args = parser.parse_args()
    
    df = generate_dataset(n_samples=args.samples, noise_ratio=args.noise, seed=args.seed, use_long_traces=args.long_traces)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / 'data'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'ci_logs_dataset.csv'
    
    df.to_csv(output_path, index=False)
    
    print(f"[OK] Dataset sauvegarde: {output_path}")
    print(f"   Taille: {len(df)} | Features: {len(df.columns) - 1}")
    print(f"   Distribution: {df['is_flaky'].value_counts().to_dict()}")


if __name__ == '__main__':
    main()
