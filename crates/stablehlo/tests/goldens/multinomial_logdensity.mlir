module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<i32>, %arg1: tensor<3xf32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.convert %arg0 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.compare LT, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.abs %arg1 : tensor<3xf32>
    %5 = stablehlo.subtract %arg1, %4 : tensor<3xf32>
    %6 = stablehlo.reshape %arg1 : (tensor<3xf32>) -> tensor<1x3xf32>
    %7 = stablehlo.reshape %5 : (tensor<3xf32>) -> tensor<1x3xf32>
    %8 = stablehlo.concatenate %6, %7, dim = 0 : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
    %9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%8 init: %9) applies stablehlo.add across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
    %11 = stablehlo.slice %10 [0:1] : (tensor<2xf32>) -> tensor<1xf32>
    %12 = stablehlo.reshape %11 : (tensor<1xf32>) -> tensor<f32>
    %13 = stablehlo.slice %10 [1:2] : (tensor<2xf32>) -> tensor<1xf32>
    %14 = stablehlo.reshape %13 : (tensor<1xf32>) -> tensor<f32>
    %15 = stablehlo.compare LT, %14, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %16 = stablehlo.subtract %12, %1 : tensor<f32>
    %17 = stablehlo.abs %16 : tensor<f32>
    %18 = stablehlo.maximum %12, %1 : tensor<f32>
    %19 = stablehlo.constant dense<0.00000286102294921875> : tensor<f32>
    %20 = stablehlo.multiply %18, %19 : tensor<f32>
    %21 = stablehlo.compare GT, %17, %20 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %22 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %23 = stablehlo.compare EQ, %12, %22 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %24 = stablehlo.select %15, %15, %21 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %25 = stablehlo.select %24, %24, %23 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %26 = stablehlo.select %2, %2, %25 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %27 = stablehlo.constant dense<0> : tensor<i32>
    %28 = stablehlo.constant dense<9> : tensor<i32>
    %29 = stablehlo.select %26, %28, %27 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %31 = stablehlo.constant dense<0.0> : tensor<3xf32>
    %32 = stablehlo.constant dense<true> : tensor<i1>
    %33 = stablehlo.constant dense<10> : tensor<i32>
    %34 = stablehlo.compare EQ, %33, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %35 = stablehlo.constant dense<"0x00000040000040400000A040"> : tensor<3xf32>
    %36 = stablehlo.constant dense<"0x00002041"> : tensor<f32>
    %37 = stablehlo.constant dense<2.0> : tensor<f32>
    %38 = stablehlo.multiply %0, %37 : tensor<f32>
    %39 = stablehlo.compare LE, %36, %38 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %40 = stablehlo.and %34, %39 : tensor<i1>
    %41 = stablehlo.and %32, %40 : tensor<i1>
    %42 = stablehlo.add %0, %1 : tensor<f32>
    %43 = chlo.lgamma %42 : tensor<f32> -> tensor<f32>
    %44 = stablehlo.constant dense<1.0> : tensor<3xf32>
    %48 = stablehlo.constant dense<"0x7DB7E8C0"> : tensor<f32>
    %49 = stablehlo.select %41, %35, %31 : (tensor<i1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    %50 = stablehlo.compare EQ, %49, %31 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    %51 = stablehlo.select %50, %44, %arg1 : (tensor<3xi1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    %52 = stablehlo.log %51 : tensor<3xf32>
    %53 = stablehlo.multiply %49, %52 : tensor<3xf32>
    %54 = stablehlo.reduce(%53 init: %9) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %55 = stablehlo.add %43, %48 : tensor<f32>
    %56 = stablehlo.add %55, %54 : tensor<f32>
    %57 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %58 = stablehlo.select %41, %56, %57 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %58, %29 : tensor<f32>, tensor<i32>
  }
}
