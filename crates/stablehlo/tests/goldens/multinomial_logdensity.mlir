module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<i32>, %arg1: tensor<3xf32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.convert %arg0 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.compare LT, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %4 = stablehlo.reduce(%arg1 init: %3) applies stablehlo.minimum across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.constant dense<0.0> : tensor<f32>
    %6 = stablehlo.compare LT, %4, %5 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg1 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %9 = stablehlo.subtract %8, %1 : tensor<f32>
    %10 = stablehlo.abs %9 : tensor<f32>
    %11 = stablehlo.maximum %8, %1 : tensor<f32>
    %12 = stablehlo.constant dense<0.00000286102294921875> : tensor<f32>
    %13 = stablehlo.multiply %11, %12 : tensor<f32>
    %14 = stablehlo.subtract %10, %13 : tensor<f32>
    %15 = stablehlo.compare GT, %14, %5 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %16 = stablehlo.compare EQ, %8, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %17 = stablehlo.select %6, %6, %15 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %18 = stablehlo.select %17, %17, %16 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %19 = stablehlo.select %2, %2, %18 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %20 = stablehlo.constant dense<0> : tensor<i32>
    %21 = stablehlo.constant dense<9> : tensor<i32>
    %22 = stablehlo.select %19, %21, %20 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %24 = stablehlo.add %0, %1 : tensor<f32>
    %25 = chlo.lgamma %24 : tensor<f32> -> tensor<f32>
    %27 = stablehlo.constant dense<"0x00000040000040400000A040"> : tensor<3xf32>
    %29 = stablehlo.constant dense<"0x1872313F6058E53F22339940"> : tensor<3xf32>
    %30 = stablehlo.reduce(%29 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %31 = stablehlo.negate %30 : tensor<f32>
    %32 = stablehlo.log %arg1 : tensor<3xf32>
    %33 = stablehlo.multiply %27, %32 : tensor<3xf32>
    %34 = stablehlo.reduce(%33 init: %7) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %35 = stablehlo.add %25, %31 : tensor<f32>
    %36 = stablehlo.add %35, %34 : tensor<f32>
    return %36, %22 : tensor<f32>, tensor<i32>
  }
}
