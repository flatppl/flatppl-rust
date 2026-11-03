module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<3x3xf32>) -> (tensor<f32>, tensor<i32>) {
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare LE, %arg0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<9> : tensor<i32>
    %5 = stablehlo.select %2, %4, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = stablehlo.cholesky %arg1, lower = true : tensor<3x3xf32>
    %7 = stablehlo.iota dim = 0 : tensor<3x3xf32>
    %8 = stablehlo.iota dim = 1 : tensor<3x3xf32>
    %9 = stablehlo.compare EQ, %7, %8 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %10 = stablehlo.constant dense<0.0> : tensor<3x3xf32>
    %11 = stablehlo.select %9, %6, %10 : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%11 init: %12) applies stablehlo.add across dimensions = [1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3xf32>
    %14 = stablehlo.log %13 : tensor<3xf32>
    %15 = stablehlo.reduce(%14 init: %12) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %16 = stablehlo.constant dense<2.0> : tensor<f32>
    %17 = stablehlo.multiply %16, %15 : tensor<f32>
    %18 = stablehlo.constant dense<1.0> : tensor<f32>
    %19 = stablehlo.subtract %arg0, %18 : tensor<f32>
    %20 = stablehlo.multiply %19, %17 : tensor<f32>
    %21 = stablehlo.multiply %16, %arg0 : tensor<f32>
    %22 = stablehlo.add %21, %1 : tensor<f32>
    %23 = stablehlo.multiply %22, %16 : tensor<f32>
    %24 = stablehlo.constant dense<0.5> : tensor<f32>
    %25 = stablehlo.add %arg0, %24 : tensor<f32>
    %26 = stablehlo.multiply %16, %25 : tensor<f32>
    %27 = chlo.lgamma %25 : tensor<f32> -> tensor<f32>
    %28 = stablehlo.multiply %16, %27 : tensor<f32>
    %29 = chlo.lgamma %26 : tensor<f32> -> tensor<f32>
    %30 = stablehlo.subtract %28, %29 : tensor<f32>
    %31 = stablehlo.multiply %16, %30 : tensor<f32>
    %32 = stablehlo.constant dense<-1.0> : tensor<f32>
    %33 = stablehlo.add %21, %32 : tensor<f32>
    %34 = stablehlo.multiply %33, %18 : tensor<f32>
    %35 = stablehlo.add %23, %34 : tensor<f32>
    %36 = stablehlo.add %arg0, %1 : tensor<f32>
    %37 = stablehlo.multiply %16, %36 : tensor<f32>
    %38 = chlo.lgamma %36 : tensor<f32> -> tensor<f32>
    %39 = stablehlo.multiply %16, %38 : tensor<f32>
    %40 = chlo.lgamma %37 : tensor<f32> -> tensor<f32>
    %41 = stablehlo.subtract %39, %40 : tensor<f32>
    %42 = stablehlo.multiply %18, %41 : tensor<f32>
    %43 = stablehlo.add %31, %42 : tensor<f32>
    %44 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %45 = stablehlo.multiply %35, %44 : tensor<f32>
    %46 = stablehlo.add %45, %43 : tensor<f32>
    %47 = stablehlo.negate %46 : tensor<f32>
    %48 = stablehlo.add %20, %47 : tensor<f32>
    return %48, %5 : tensor<f32>, tensor<i32>
  }
}
